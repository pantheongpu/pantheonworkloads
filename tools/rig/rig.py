#!/usr/bin/env python3
"""Rented-rig tooling (docs/rigs.md): launch, teardown, status, extend. The .sh files beside this one call it.

    rig.py launch   --tag T --instance-type X [--ami-name GLOB | --ami ID] [--disk-gib N] [--azs a,b] --max-hours H
                    --max-dollars D [--rate USD_PER_H] [--region R] [--dry-run]
    rig.py teardown TAG
    rig.py status
    rig.py extend   TAG HOURS --reason TEXT [--max-dollars D]

Everything the tooling knows about a rig is in ~/.cache/pw-rigs/<tag>/ (rig.json and the ssh key, mode 0600), never in a
scratchpad or /tmp; the audit trail is ~/.cache/pw-rigs/ledger.tsv. Environment, for tests: PW_AWS (the aws command),
PW_SSH / PW_SCP, PW_RIGS_HOME, PW_MY_IP, PW_RIG_NOW (epoch seconds), PW_RIG_NOSLEEP=1, PW_RIGS_ALLOW_UNSAFE_HOME=1.
No call here uses anything but the aws CLI and ssh.
"""
import argparse
import datetime
import fcntl
import json
import os
import pathlib
import re
import subprocess
import sys
import time
import urllib.request

# ---------------------------------------------------------------- rates and caps

# On-demand Linux $/hour, us-east-1. WRITTEN FROM MEMORY OF THE LIST PRICES, NOT READ FROM AWS: they are the higher, older
# list prices for the p4/p5 classes on purpose (a rate that is too high ends a rig early, one that is too low overspends).
# Pass --rate to override; launch prints which source it used.
RATES = {
    "g5.xlarge": 1.006, "g5.2xlarge": 1.212, "g5.4xlarge": 1.624, "g5.12xlarge": 5.672, "g5.48xlarge": 16.288,
    "g6.xlarge": 0.8048, "g6e.xlarge": 1.861, "g6e.12xlarge": 10.493,
    "p4d.24xlarge": 32.77, "p4de.24xlarge": 40.96, "p5.48xlarge": 98.32,
    "c7i.2xlarge": 0.357, "c7i.4xlarge": 0.714, "c6i.4xlarge": 0.68, "m7i.4xlarge": 0.8064, "r6i.2xlarge": 0.504,
}
# EBS gp3, $/GiB-month, $/IOPS-month above 3000, $/MiB/s-month above 125 (from memory; used for the estimate only)
GP3_GIB, GP3_IOPS, GP3_TP = 0.08, 0.005, 0.04
HOURS_PER_MONTH = 730.0
DEADLINE_GRACE_S = 600       # status: the on-box timer fires at the deadline; this much later is "over"
DEFAULT_AMI_NAME = "Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04) *"
TAG_RE = re.compile(r"^pw-[a-z0-9][a-z0-9-]{0,40}$")
TOOL_TAG = "pw-rig-tool"


class RigError(Exception):
    pass


def ebs_hourly(gib, iops=3000, throughput=125):
    month = gib * GP3_GIB + max(0, iops - 3000) * GP3_IOPS + max(0, throughput - 125) * GP3_TP
    return month / HOURS_PER_MONTH


def hourly_cost(rate, gib=0, iops=3000, throughput=125):
    return rate + ebs_hourly(gib, iops, throughput)


def caps(max_hours, max_dollars, hourly):
    """(hours, why): the rig's lifetime is the smaller of max_hours and what max_dollars buys at `hourly` $/h."""
    if max_hours <= 0 or max_dollars <= 0 or hourly <= 0:
        raise RigError("max hours, max dollars and the hourly rate must all be positive")
    by_dollars = max_dollars / hourly
    if by_dollars < max_hours:
        return by_dollars, f"max-dollars ${max_dollars:g} at ${hourly:.3f}/h"
    return float(max_hours), f"max-hours {max_hours:g}"


def cost_usd(seconds, hourly):
    return max(0.0, seconds) / 3600.0 * hourly


# ---------------------------------------------------------------- time, state, ledger

def now():
    return float(os.environ.get("PW_RIG_NOW") or time.time())


def iso(ts):
    return datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sleep(s):
    if not os.environ.get("PW_RIG_NOSLEEP"):
        time.sleep(s)


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def check_home(path):
    """The rigs directory must not be a scratchpad or temp dir shared with other agents (a key was once lost to a
    scratchpad collision). Refuses any path with a 'scratchpad' component or under /tmp/claude-*."""
    p = pathlib.Path(path).expanduser().resolve()
    bad = "scratchpad" in p.parts or any(part.startswith("claude-") and i > 0 and p.parts[i - 1] == "tmp"
                                          for i, part in enumerate(p.parts))
    if bad and os.environ.get("PW_RIGS_ALLOW_UNSAFE_HOME") != "1":
        raise RigError(f"{p} looks like a shared scratchpad; keys and state must live in a private directory "
                       f"(default ~/.cache/pw-rigs). Unset PW_RIGS_HOME.")
    return p


def home():
    h = check_home(os.environ.get("PW_RIGS_HOME") or "~/.cache/pw-rigs")
    h.mkdir(parents=True, exist_ok=True, mode=0o700)
    return h


def rig_dir(tag):
    return home() / tag


def load_rig(tag):
    f = rig_dir(tag) / "rig.json"
    if not f.is_file():
        return None
    return json.loads(f.read_text())


def save_rig(rig):
    d = rig_dir(rig["tag"])
    d.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = d / "rig.json.tmp"
    tmp.write_text(json.dumps(rig, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, d / "rig.json")


def ledger(tag, event, **fields):
    """Append one line: time, tag, event, key=value ... (tabs and newlines in values become spaces)."""
    f = home() / "ledger.tsv"
    new = not f.exists()
    flat = re.compile(r"[\t\r\n]+")
    cells = [iso(now()), tag, event] + [k + "=" + flat.sub(" ", str(v)) for k, v in fields.items()]
    with open(f, "a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        if new:
            fh.write("time\ttag\tevent\tdetails\n")
        fh.write("\t".join(cells) + "\n")


def norm_tag(tag):
    tag = tag if tag.startswith("pw-") else "pw-" + tag
    if not TAG_RE.match(tag):
        raise RigError(f"bad tag {tag!r}: pw- then lower-case letters, digits, '-' (max 44 characters)")
    return tag


# ---------------------------------------------------------------- aws and ssh

class AwsError(Exception):
    def __init__(self, args, code, message):
        super().__init__(f"aws {' '.join(args[:2])}: {code}: {message}")
        self.code, self.message = code, message


def aws(args, region):
    cmd = [os.environ.get("PW_AWS") or "aws", *args, "--region", region, "--output", "json"]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        raise RigError(f"the aws CLI ({cmd[0]}) is not installed or not on PATH")
    if p.returncode != 0:
        m = re.search(r"An error occurred \((\w[\w.]*)\)[^:]*: ?(.*)", p.stderr or "")
        raise AwsError(args, m.group(1) if m else "Error", (m.group(2) if m else (p.stderr or "").strip())[:300])
    return json.loads(p.stdout) if p.stdout.strip() else {}


def tagspec(rtype, tags):
    return {"ResourceType": rtype, "Tags": [{"Key": k, "Value": str(v)} for k, v in tags.items()]}


def ssh_base(rig, tty=False):
    d = rig_dir(rig["tag"])
    return [os.environ.get("PW_SSH") or "ssh", "-i", str(d / "key.pem"), "-o", "IdentitiesOnly=yes",
            "-o", "StrictHostKeyChecking=accept-new", "-o", f"UserKnownHostsFile={d / 'known_hosts'}",
            "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=30", f"{rig['user']}@{rig['ip']}"]


def ssh_run(rig, remote, timeout=None):
    return subprocess.run([*ssh_base(rig), remote], capture_output=True, text=True, timeout=timeout)


def connection_command(rig):
    return " ".join(["ssh", "-i", str(rig_dir(rig["tag"]) / "key.pem"), "-o", "IdentitiesOnly=yes", f"{rig['user']}@{rig['ip']}"])


def public_ip():
    ip = os.environ.get("PW_MY_IP")
    if ip:
        return ip
    with urllib.request.urlopen("https://checkip.amazonaws.com", timeout=15) as r:
        ip = r.read().decode().strip()
    if not re.match(r"^\d+\.\d+\.\d+\.\d+$", ip):
        raise RigError(f"cannot determine this machine's public IP (got {ip!r}); pass --my-ip")
    return ip


# ---------------------------------------------------------------- the on-box timer

WATCHDOG = r'''#!/bin/bash
# pw-rig watchdog: terminates this instance (shutdown behaviour = terminate) at the absolute deadline in
# /etc/pw-rig/deadline (epoch seconds). Fails closed: a missing or unreadable deadline file counts as expired.
f=/etc/pw-rig/deadline
d=$(cat "$f" 2>/dev/null || true)
if ! [[ "$d" =~ ^[0-9]+$ ]]; then logger -t pw-rig "deadline file missing or unreadable: terminating"; d=0; fi
left=$(( d - $(date +%s) ))
if (( left <= 0 )); then
  logger -t pw-rig "deadline reached: terminating"
  wall "pw-rig: deadline reached, this instance terminates now" 2>/dev/null || true
  /sbin/shutdown -h now "pw-rig deadline reached"
  exit 0
fi
if (( left <= 900 )); then wall "pw-rig: this instance terminates in $(( left / 60 )) min (extend with tools/rig/extend.sh from the launching machine)" 2>/dev/null || true; fi
'''


def user_data(tag, deadline):
    """cloud-init script: writes the root-owned deadline file and a systemd timer that enforces it every minute.
    The files are made immutable (chattr +i) so a plain edit, rm or `systemctl disable` by an agent fails; only
    root can undo that on purpose, and extend.sh does it from outside and logs to the ledger."""
    return f'''#!/bin/bash
set -eu
mkdir -p /etc/pw-rig
printf '%s\\n' {int(deadline)} > /etc/pw-rig/deadline
printf '%s\\n' {tag} > /etc/pw-rig/tag
cat > /usr/local/sbin/pw-rig-watchdog <<'PW_EOF'
{WATCHDOG}PW_EOF
cat > /etc/systemd/system/pw-rig-watchdog.service <<'PW_EOF'
[Unit]
Description=pw-rig absolute-deadline watchdog
[Service]
Type=oneshot
ExecStart=/usr/local/sbin/pw-rig-watchdog
PW_EOF
cat > /etc/systemd/system/pw-rig-watchdog.timer <<'PW_EOF'
[Unit]
Description=pw-rig absolute-deadline watchdog, every minute
[Timer]
OnBootSec=30s
OnUnitActiveSec=60s
AccuracySec=5s
[Install]
WantedBy=timers.target
PW_EOF
chown root:root /etc/pw-rig/deadline /etc/pw-rig/tag /usr/local/sbin/pw-rig-watchdog
chmod 0644 /etc/pw-rig/deadline /etc/pw-rig/tag
chmod 0755 /usr/local/sbin/pw-rig-watchdog
systemctl daemon-reload
systemctl enable --now pw-rig-watchdog.timer
chattr +i /etc/pw-rig/deadline /etc/pw-rig/tag /usr/local/sbin/pw-rig-watchdog \\
  /etc/systemd/system/pw-rig-watchdog.service /etc/systemd/system/pw-rig-watchdog.timer
'''


def remote_set_deadline(deadline):
    return ("sudo chattr -i /etc/pw-rig/deadline && "
            f"echo {int(deadline)} | sudo tee /etc/pw-rig/deadline >/dev/null && "
            "sudo chattr +i /etc/pw-rig/deadline && cat /etc/pw-rig/deadline")


def verify_timer(rig, wait_s):
    """Wait up to wait_s for ssh, then check the deadline file on the box holds the deadline we set. Returns an error
    string, or None when the timer is installed and agrees."""
    end = time.time() + wait_s
    last = ""
    while True:
        try:
            p = ssh_run(rig, "cat /etc/pw-rig/deadline; systemctl is-active pw-rig-watchdog.timer", timeout=60)
            last = (p.stdout + p.stderr).strip()
            lines = p.stdout.split()
            if p.returncode == 0 and lines and lines[0] == str(int(rig["deadline"])) and lines[-1] == "active":
                return None
            if p.returncode == 0 and lines and lines[0] != str(int(rig["deadline"])):
                return f"deadline file on the box says {lines[0]}, expected {int(rig['deadline'])}"
        except (subprocess.TimeoutExpired, OSError) as e:
            last = str(e)
        if time.time() >= end:
            return f"could not verify the timer over ssh in {wait_s:g} s (last: {last[:150]})"
        sleep(10)


# ---------------------------------------------------------------- launch

def cmd_launch(a):
    tag = norm_tag(a.tag)
    region = a.region
    rate, source = (a.rate, "--rate") if a.rate else (RATES.get(a.instance_type), "built-in table (unverified list price from memory)")
    if rate is None:
        raise RigError(f"no hourly rate for {a.instance_type}: pass --rate USD_PER_HOUR (table has {', '.join(sorted(RATES))})")
    hourly = hourly_cost(rate, a.disk_gib, a.iops, a.throughput)
    hours, why = caps(a.max_hours, a.max_dollars, hourly)
    azs = [x for x in a.azs.split(",") if x] if a.azs else [f"{region}{s}" for s in "abcdf"]
    plan = (f"{tag}: {a.instance_type} in {region} (AZs tried in order: {', '.join(azs)}), {a.disk_gib} GiB gp3 "
            f"{a.iops} IOPS {a.throughput} MiB/s, rate ${rate:g}/h from {source}, with disk ${hourly:.3f}/h; "
            f"lives {hours:.2f} h (limited by {why}); worst case ${hours * hourly:.2f}")
    if a.dry_run:
        print("DRY RUN, no aws call made")
        print(plan)
        print(user_data(tag, now() + hours * 3600))
        return 0
    d = rig_dir(tag)
    if (d / "rig.json").exists():
        raise RigError(f"{d}/rig.json exists: a rig called {tag} was launched from this machine before. Pick another --tag "
                       f"or run teardown.sh {tag} first (refusing to overwrite its key)")
    my_ip = a.my_ip or public_ip()
    log(plan)

    # AMI
    if a.ami:
        ami = a.ami
    else:
        imgs = aws(["ec2", "describe-images", "--owners", "amazon", "--filters", f"Name=name,Values={a.ami_name}",
                    "Name=state,Values=available"], region).get("Images", [])
        if not imgs:
            raise RigError(f"no AMI matches {a.ami_name!r} in {region}")
        ami = max(imgs, key=lambda i: i.get("CreationDate", ""))["ImageId"]
        log(f"AMI {ami} ({max(imgs, key=lambda i: i.get('CreationDate', '')).get('Name')})")

    # default subnets, one per AZ
    subs = aws(["ec2", "describe-subnets", "--filters", "Name=default-for-az,Values=true"], region).get("Subnets", [])
    by_az = {s["AvailabilityZone"]: s for s in subs}
    usable = [z for z in azs if z in by_az]
    if not usable:
        raise RigError(f"none of {azs} has a default subnet (have {sorted(by_az)})")

    d.mkdir(parents=True, mode=0o700)
    rig = {"tag": tag, "region": region, "instance_type": a.instance_type, "ami": ami, "user": a.user, "state": "launching",
           "rate": rate, "hourly": hourly, "max_hours": a.max_hours, "max_dollars": a.max_dollars, "my_ip": my_ip,
           "disk_gib": a.disk_gib, "created": now(), "key_name": tag, "extensions": []}
    created_key = created_sg = False
    try:
        key = aws(["ec2", "create-key-pair", "--key-name", tag, "--key-type", "ed25519", "--key-format", "pem"], region)
        created_key = True
        fd = os.open(d / "key.pem", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as fh:
            fh.write(key["KeyMaterial"])
        os.chmod(d / "key.pem", 0o600)
        rig["key_created"] = True
        save_rig(rig)

        vpc = by_az[usable[0]]["VpcId"]
        sg = aws(["ec2", "create-security-group", "--group-name", tag, "--description", f"pw-rig {tag}: ssh from {my_ip} only",
                  "--vpc-id", vpc, "--tag-specifications", json.dumps([tagspec("security-group", {"Name": tag, TOOL_TAG: 1})])], region)
        created_sg = True
        rig["sg_id"] = sg["GroupId"]
        save_rig(rig)
        aws(["ec2", "authorize-security-group-ingress", "--group-id", sg["GroupId"], "--protocol", "tcp", "--port", "22",
             "--cidr", f"{my_ip}/32"], region)

        inst, tried = None, []
        for az in usable:
            deadline = now() + hours * 3600
            tags = {"Name": tag, TOOL_TAG: 1, "pw-deadline": int(deadline), "pw-deadline-iso": iso(deadline),
                    "pw-max-dollars": a.max_dollars, "pw-hourly-usd": f"{hourly:.4f}", "pw-owner": os.environ.get("USER", "unknown")}
            bdm = [{"DeviceName": a.root_device, "Ebs": {"VolumeSize": a.disk_gib, "VolumeType": "gp3", "Iops": a.iops,
                                                          "Throughput": a.throughput, "DeleteOnTermination": True}}]
            try:
                r = aws(["ec2", "run-instances", "--image-id", ami, "--instance-type", a.instance_type, "--count", "1",
                         "--key-name", tag, "--security-group-ids", sg["GroupId"], "--subnet-id", by_az[az]["SubnetId"],
                         "--instance-initiated-shutdown-behavior", "terminate",
                         "--metadata-options", "HttpTokens=required,HttpEndpointEnabled=enabled",
                         "--block-device-mappings", json.dumps(bdm),
                         "--tag-specifications", json.dumps([tagspec("instance", tags), tagspec("volume", {"Name": tag, TOOL_TAG: 1})]),
                         "--user-data", user_data(tag, deadline)], region)
            except AwsError as e:
                if e.code in ("InsufficientInstanceCapacity", "Unsupported"):
                    log(f"{az}: {e.code}, trying the next AZ")
                    tried.append(f"{az}: {e.code}")
                    continue
                raise
            inst = r["Instances"][0]
            rig.update(instance_id=inst["InstanceId"], az=az, deadline=deadline, state="pending", launched=now())
            save_rig(rig)
            break
        if inst is None:
            raise RigError("no capacity: " + "; ".join(tried))
    except BaseException as e:
        # nothing is running (or run-instances itself failed): release what this call created
        log(f"launch failed ({e}); releasing what was created")
        if rig.get("instance_id"):
            raise
        ledger(tag, "launch-failed", detail=str(e)[:200])
        _release(rig, region, created_key, created_sg)
        shutil_rmtree(d)
        raise

    ledger(tag, "launch", instance=rig["instance_id"], type=a.instance_type, az=rig["az"], region=region, hourly=f"{hourly:.4f}",
           max_hours=a.max_hours, max_dollars=a.max_dollars, deadline=iso(rig["deadline"]), hours=f"{hours:.2f}", why=why)

    # wait for running + address
    end = time.time() + a.boot_wait
    while True:
        res = aws(["ec2", "describe-instances", "--instance-ids", rig["instance_id"]], region)
        i = res["Reservations"][0]["Instances"][0]
        if i["State"]["Name"] == "running" and i.get("PublicIpAddress"):
            rig["ip"], rig["state"] = i["PublicIpAddress"], "running"
            if i.get("LaunchTime"):
                rig["launched"] = _parse_time(i["LaunchTime"]) or rig["launched"]
            save_rig(rig)
            break
        if i["State"]["Name"] in ("terminated", "shutting-down") or time.time() > end:
            raise RigError(f"{rig['instance_id']} is {i['State']['Name']} and has no address after {a.boot_wait:g} s; "
                           f"run teardown.sh {tag}")
        sleep(5)
    print(f"launched {rig['instance_id']} ({a.instance_type}, {rig['az']}); terminates itself at {iso(rig['deadline'])}")
    print(f"connect: {connection_command(rig)}")
    print(f"key:     {d / 'key.pem'}")
    print(f"extend:  tools/rig/extend.sh {tag} <hours> --reason '...'    done: tools/rig/teardown.sh {tag}")
    if a.ssh_wait > 0:
        problem = verify_timer(rig, a.ssh_wait)
        if problem:
            print(f"TIMER NOT VERIFIED: {problem}. The rig is up and billing; fix it or run teardown.sh {tag}.", file=sys.stderr)
            ledger(tag, "timer-unverified", detail=problem)
            return 3
        print("timer verified: /etc/pw-rig/deadline on the box holds the deadline and the watchdog timer is active")
    return 0


def shutil_rmtree(d):
    import shutil
    shutil.rmtree(d, ignore_errors=True)


def _parse_time(text):
    try:
        return datetime.datetime.fromisoformat(str(text).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _release(rig, region, key, sg):
    if sg and rig.get("sg_id"):
        try:
            aws(["ec2", "delete-security-group", "--group-id", rig["sg_id"]], region)
        except AwsError as e:
            log(f"could not delete security group {rig['sg_id']}: {e}")
    if key:
        try:
            aws(["ec2", "delete-key-pair", "--key-name", rig["tag"]], region)
        except AwsError as e:
            log(f"could not delete key pair {rig['tag']}: {e}")


# ---------------------------------------------------------------- teardown

def _instances(region, tag):
    res = aws(["ec2", "describe-instances", "--filters", f"Name=tag:Name,Values={tag}"], region)
    return [i for r in res.get("Reservations", []) for i in r.get("Instances", [])]


def _terminated_at(inst):
    m = re.search(r"\((\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) GMT\)", inst.get("StateTransitionReason") or "")
    if m:
        return datetime.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=datetime.timezone.utc).timestamp()
    return None


def cmd_teardown(a):
    tag = norm_tag(a.tag)
    rig = load_rig(tag)
    region = a.region or (rig or {}).get("region") or os.environ.get("AWS_REGION") or "us-east-1"
    insts = _instances(region, tag)
    if rig is None and not insts:
        print(f"{tag}: nothing known locally and nothing in {region}; nothing to do")
        return 0
    for i in insts:
        tags = {t["Key"]: t["Value"] for t in i.get("Tags", [])}
        if tags.get(TOOL_TAG) != "1" and not a.force_foreign:
            raise RigError(f"{i['InstanceId']} is named {tag} but was not launched by tools/rig (no {TOOL_TAG} tag): "
                           f"it may belong to someone else. Refusing; --force-foreign overrides.")
    live = [i for i in insts if i["State"]["Name"] != "terminated"]
    if live:
        aws(["ec2", "terminate-instances", "--instance-ids", *[i["InstanceId"] for i in live]], region)
        log(f"terminating {', '.join(i['InstanceId'] for i in live)}")
    end = time.time() + a.wait
    while True:
        insts = _instances(region, tag)
        if all(i["State"]["Name"] == "terminated" for i in insts):
            break
        if time.time() > end:
            raise RigError(f"still not terminated after {a.wait:g} s: " + ", ".join(f"{i['InstanceId']}={i['State']['Name']}" for i in insts))
        sleep(5)
    problems = []

    # volumes: none may be left (DeleteOnTermination removes them; give it time, then delete any that stayed)
    end = time.time() + a.wait
    while True:
        vols = aws(["ec2", "describe-volumes", "--filters", f"Name=tag:Name,Values={tag}"], region).get("Volumes", [])
        if not vols or time.time() > end:
            break
        sleep(5)
    for v in vols:
        if v["State"] == "available":
            try:
                aws(["ec2", "delete-volume", "--volume-id", v["VolumeId"]], region)
                log(f"deleted leftover volume {v['VolumeId']}")
            except AwsError as e:
                log(f"could not delete {v['VolumeId']}: {e}")
    for v in aws(["ec2", "describe-volumes", "--filters", f"Name=tag:Name,Values={tag}"], region).get("Volumes", []):
        problems.append(f"volume {v['VolumeId']} is still {v['State']}")

    # key pair, security group (idempotent: a missing one is fine)
    try:
        aws(["ec2", "delete-key-pair", "--key-name", tag], region)
    except AwsError as e:
        if "NotFound" not in e.code:
            problems.append(str(e))
    sgs = aws(["ec2", "describe-security-groups", "--filters", f"Name=group-name,Values={tag}"], region).get("SecurityGroups", [])
    for sg in sgs:
        for attempt in range(24):
            try:
                aws(["ec2", "delete-security-group", "--group-id", sg["GroupId"]], region)
                break
            except AwsError as e:
                if "NotFound" in e.code:
                    break
                if e.code == "DependencyViolation" and attempt < 23:
                    sleep(5)
                    continue
                problems.append(str(e))
                break
    if aws(["ec2", "describe-security-groups", "--filters", f"Name=group-name,Values={tag}"], region).get("SecurityGroups"):
        problems.append(f"security group {tag} is still there")

    # local: the private key is useless now
    key = rig_dir(tag) / "key.pem"
    if key.exists():
        key.unlink()
    if rig is not None and not rig.get("teardown_logged") and not problems:
        end_ts = max([t for t in (_terminated_at(i) for i in insts) if t] or [now()])
        end_ts = min(end_ts, now())
        started = rig.get("launched") or rig.get("created") or end_ts
        hours = max(0.0, end_ts - started) / 3600
        usd = cost_usd(end_ts - started, rig.get("hourly", 0))
        ledger(tag, "teardown", instance=rig.get("instance_id", ""), hours=f"{hours:.2f}", usd=f"{usd:.2f}",
               cap_usd=rig.get("max_dollars", ""), estimate="elapsed x (list rate + disk), not a bill")
        rig.update(state="terminated", ended=end_ts, cost_usd=round(usd, 2), teardown_logged=True)
        save_rig(rig)
        print(f"{tag}: torn down; {hours:.2f} h, estimated ${usd:.2f} (cap ${rig.get('max_dollars', '?')})")
    elif rig is not None and rig.get("teardown_logged"):
        print(f"{tag}: already torn down earlier (estimated ${rig.get('cost_usd')}); AWS re-checked")
    if problems:
        for p in problems:
            print(f"PROBLEM: {p}", file=sys.stderr)
        return 1
    print(f"{tag}: verified terminated, no volumes, key pair and security group gone")
    return 0


# ---------------------------------------------------------------- status

def cmd_status(a):
    regions = a.region.split(",")
    t = now()
    rows, bad = [], False
    seen_tags = set()
    for region in regions:
        res = aws(["ec2", "describe-instances", "--filters", "Name=tag:Name,Values=pw-*",
                   "Name=instance-state-name,Values=pending,running,stopping,stopped,shutting-down"], region)
        for r in res.get("Reservations", []):
            for i in r.get("Instances", []):
                tags = {x["Key"]: x["Value"] for x in i.get("Tags", [])}
                tag = tags.get("Name", i["InstanceId"])
                seen_tags.add(tag)
                rig = load_rig(tag) if TAG_RE.match(tag) else None
                launch = _parse_time(i.get("LaunchTime")) or (rig or {}).get("launched") or t
                hourly = (rig or {}).get("hourly") or _f(tags.get("pw-hourly-usd")) or RATES.get(i.get("InstanceType"), 0)
                cap = (rig or {}).get("max_dollars") or _f(tags.get("pw-max-dollars"))
                deadline = (rig or {}).get("deadline") or _f(tags.get("pw-deadline"))
                elapsed = (t - launch) / 3600
                cost = cost_usd(t - launch, hourly)
                flags = []
                if cap is None or deadline is None:
                    flags.append("NO-CAP-RECORD")
                if cap is not None and cost > cap:
                    flags.append("OVER-CAP")
                if deadline is not None and t > deadline + DEADLINE_GRACE_S:
                    flags.append("PAST-DEADLINE")
                if i["State"]["Name"] == "stopped":
                    flags.append("STOPPED-EBS-BILLING")
                if tags.get(TOOL_TAG) != "1":
                    flags.append("FOREIGN")
                rows.append((tag, i["InstanceType"], i["State"]["Name"], elapsed, (deadline - t) / 3600 if deadline else None,
                             cost, cap, flags, region))
                bad = bad or any(f in ("NO-CAP-RECORD", "OVER-CAP", "PAST-DEADLINE") for f in flags)
        vols = aws(["ec2", "describe-volumes", "--filters", "Name=tag:Name,Values=pw-*"], region).get("Volumes", [])
        for v in vols:
            tag = {x["Key"]: x["Value"] for x in v.get("Tags", [])}.get("Name", "?")
            orphan = v["State"] == "available"
            rows.append((tag, f"volume {v['VolumeId']} {v['Size']} GiB", v["State"], None, None, None, None,
                         ["ORPHAN-VOLUME"] if orphan else [], region))
            bad = bad or orphan
    print(f"{'TAG':24} {'TYPE':22} {'STATE':12} {'HOURS':>7} {'LEFT_H':>7} {'COST$':>8} {'CAP$':>7}  FLAGS")
    for tag, typ, state, el, left, cost, cap, flags, _ in rows:
        print(f"{tag:24} {typ:22} {state:12} {_n(el):>7} {_n(left):>7} {_n(cost):>8} {_n(cap):>7}  {' '.join(flags)}")
    if not rows:
        print("(no pw-* instances or volumes)")
    # local records that say running but AWS shows nothing: informational
    h = home()
    for d in sorted(h.iterdir()):
        rig = load_rig(d.name) if d.is_dir() and TAG_RE.match(d.name) else None
        if rig and rig.get("state") in ("running", "pending") and rig["tag"] not in seen_tags:
            print(f"note: {rig['tag']} is recorded as {rig['state']} here but AWS shows no live instance; run teardown.sh {rig['tag']} to close it out")
    print("OVER A CAP, A DEADLINE, OR LEAKING" if bad else "ok")
    return 1 if bad else 0


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _n(x):
    return "" if x is None else f"{x:.2f}"


# ---------------------------------------------------------------- extend

def cmd_extend(a):
    tag = norm_tag(a.tag)
    rig = load_rig(tag)
    if rig is None or rig.get("state") != "running":
        raise RigError(f"{tag} is not a running rig launched from this machine")
    if a.hours <= 0:
        raise RigError("hours must be positive")
    if len((a.reason or "").strip()) < 10:
        raise RigError("--reason must say why, in at least 10 characters; it goes in the ledger")
    t = now()
    if t > rig["deadline"]:
        raise RigError(f"{tag}'s deadline ({iso(rig['deadline'])}) has passed; the box is terminating. Launch a new rig.")
    new_deadline = rig["deadline"] + a.hours * 3600
    cap = a.max_dollars or rig["max_dollars"]
    projected = cost_usd(new_deadline - (rig.get("launched") or rig["created"]), rig["hourly"])
    if projected > cap:
        raise RigError(f"extending by {a.hours:g} h projects ${projected:.2f} at ${rig['hourly']:.3f}/h, over the cap of ${cap:g}. "
                       f"Pass --max-dollars N to raise the cap (it is logged).")
    p = ssh_run(rig, remote_set_deadline(new_deadline), timeout=90)
    if p.returncode != 0 or p.stdout.split()[-1:] != [str(int(new_deadline))]:
        raise RigError(f"could not set the deadline on the box (ssh exit {p.returncode}): {(p.stderr or p.stdout).strip()[-200:]}")
    old = rig["deadline"]
    rig["extensions"] = rig.get("extensions", []) + [{"at": t, "hours": a.hours, "reason": a.reason, "from": old, "to": new_deadline}]
    rig.update(deadline=new_deadline, max_dollars=cap)
    save_rig(rig)
    try:
        aws(["ec2", "create-tags", "--resources", rig["instance_id"], "--tags", f"Key=pw-deadline,Value={int(new_deadline)}",
             f"Key=pw-deadline-iso,Value={iso(new_deadline)}", f"Key=pw-max-dollars,Value={cap:g}"], rig["region"])
    except AwsError as e:
        log(f"warning: instance tags not updated ({e}); the box and rig.json are")
    ledger(tag, "extend", hours=a.hours, old=iso(old), new=iso(new_deadline), projected_usd=f"{projected:.2f}", cap_usd=f"{cap:g}",
           reason=a.reason)
    print(f"{tag}: deadline {iso(old)} -> {iso(new_deadline)}; projected worst case ${projected:.2f} of ${cap:g}")
    return 0


# ---------------------------------------------------------------- cli

def build_parser():
    ap = argparse.ArgumentParser(prog="rig.py")
    ap.add_argument("--region-default", help=argparse.SUPPRESS)
    sub = ap.add_subparsers(dest="cmd", required=True)
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
    l = sub.add_parser("launch")
    l.add_argument("--tag", required=True, help="name; 'pw-' is added when missing")
    l.add_argument("--instance-type", required=True)
    l.add_argument("--region", default=region)
    l.add_argument("--azs", help="comma list tried in order on InsufficientInstanceCapacity (default <region>a,b,c,d,f)")
    l.add_argument("--ami", help="an AMI id (skips the name search)")
    l.add_argument("--ami-name", default=DEFAULT_AMI_NAME, help="name glob searched among Amazon's AMIs, newest wins")
    l.add_argument("--user", default="ubuntu")
    l.add_argument("--root-device", default="/dev/sda1")
    l.add_argument("--disk-gib", type=int, default=600)
    l.add_argument("--iops", type=int, default=3000)
    l.add_argument("--throughput", type=int, default=125, help="MiB/s, gp3")
    l.add_argument("--max-hours", type=float, required=True)
    l.add_argument("--max-dollars", type=float, required=True)
    l.add_argument("--rate", type=float, help="on-demand $/hour (overrides the built-in table)")
    l.add_argument("--my-ip", help="the address ssh is allowed from (default: looked up)")
    l.add_argument("--ssh-wait", type=float, default=300, help="seconds to wait for ssh and verify the timer (0: do not)")
    l.add_argument("--boot-wait", type=float, default=600)
    l.add_argument("--dry-run", action="store_true")
    l.set_defaults(fn=cmd_launch)
    t = sub.add_parser("teardown")
    t.add_argument("tag")
    t.add_argument("--region")
    t.add_argument("--wait", type=float, default=900)
    t.add_argument("--force-foreign", action="store_true")
    t.set_defaults(fn=cmd_teardown)
    s = sub.add_parser("status")
    s.add_argument("--region", default=region, help="comma list")
    s.set_defaults(fn=cmd_status)
    e = sub.add_parser("extend")
    e.add_argument("tag")
    e.add_argument("hours", type=float)
    e.add_argument("--reason", required=True)
    e.add_argument("--max-dollars", type=float)
    e.set_defaults(fn=cmd_extend)
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    try:
        return a.fn(a)
    except RigError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except (AwsError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
