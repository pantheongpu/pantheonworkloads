"""Tests for tools/rig (launch, teardown, status, extend, the cap math) driven against a fake `aws` and a fake `ssh`
(tests/rig_fakes). No AWS call is made, nothing is launched; light enough for any machine.

    python3 -m unittest tests/test_rig.py -v
"""
import contextlib
import io
import json
import os
import pathlib
import re
import shutil
import stat
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
FAKES = ROOT / "tests" / "rig_fakes"
sys.path.insert(0, str(ROOT / "tools" / "rig"))
import rig  # noqa: E402

NOW = 1_800_000_000.0


class RigCase(unittest.TestCase):
    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp(prefix="rigtest-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        env = {
            "PW_AWS": str(FAKES / "aws"), "PW_SSH": str(FAKES / "ssh"), "PW_SCP": str(FAKES / "scp"),
            "FAKE_AWS_STATE": str(self.tmp / "aws.json"), "FAKE_S3_ROOT": str(self.tmp / "s3"),
            "FAKE_SSH_LOG": str(self.tmp / "ssh.log"), "FAKE_BOX_DIR": str(self.tmp / "box"),
            "PW_RIGS_HOME": str(self.tmp / "rigs"), "PW_RIGS_ALLOW_UNSAFE_HOME": "1",
            "PW_MY_IP": "198.51.100.7", "PW_RIG_NOSLEEP": "1", "PW_RIG_NOW": str(NOW), "HOME": str(self.tmp / "home"),
        }
        p = mock.patch.dict(os.environ, env)
        p.start()
        self.addCleanup(p.stop)
        for k in ("FAKE_SSH_DOWN", "AWS_REGION", "AWS_DEFAULT_REGION"):
            os.environ.pop(k, None)
        (self.tmp / "home").mkdir()
        self.state_file = self.tmp / "aws.json"
        self.set_state(now=NOW)

    # -- helpers
    def set_state(self, **kw):
        s = json.loads(self.state_file.read_text()) if self.state_file.exists() else {}
        s.update(kw)
        self.state_file.write_text(json.dumps(s))

    def state(self):
        return json.loads(self.state_file.read_text())

    def calls(self):
        f = pathlib.Path(str(self.state_file) + ".calls")
        return [json.loads(x) for x in f.read_text().splitlines()] if f.exists() else []

    def called(self, op):
        return [c for c in self.calls() if c[1] == op]

    def run_rig(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = rig.main(list(argv))
        return rc, out.getvalue(), err.getvalue()

    def launch(self, tag="pw-t1", extra=(), rate="10", hours="8", dollars="1000", disk="100"):
        return self.run_rig("launch", "--tag", tag, "--instance-type", "p4de.24xlarge", "--rate", rate, "--max-hours", hours,
                            "--max-dollars", dollars, "--disk-gib", disk, "--ssh-wait", "5", *extra)

    def ledger_lines(self):
        f = self.tmp / "rigs" / "ledger.tsv"
        return f.read_text().splitlines() if f.exists() else []


class TestCapMath(unittest.TestCase):
    def test_hours_limit_wins_when_dollars_are_plentiful(self):
        h, why = rig.caps(8, 1000, 10)
        self.assertEqual(h, 8.0)
        self.assertIn("max-hours", why)

    def test_dollar_limit_wins_when_it_buys_fewer_hours(self):
        h, why = rig.caps(8, 20, 10)
        self.assertAlmostEqual(h, 2.0)
        self.assertIn("max-dollars", why)

    def test_nonpositive_inputs_are_refused(self):
        for args in [(0, 10, 1), (1, 0, 1), (1, 10, 0), (-1, 10, 1)]:
            with self.assertRaises(rig.RigError):
                rig.caps(*args)

    def test_disk_adds_to_the_hourly_cost(self):
        self.assertAlmostEqual(rig.ebs_hourly(730 / 0.08, 3000, 125) * 730, 730, places=6)  # $730/month of gp3 storage
        self.assertEqual(rig.ebs_hourly(0), 0)
        self.assertGreater(rig.ebs_hourly(100, 16000, 1000), rig.ebs_hourly(100))
        self.assertAlmostEqual(rig.hourly_cost(10, 0), 10)

    def test_cost_is_elapsed_times_rate(self):
        self.assertAlmostEqual(rig.cost_usd(5400, 10), 15.0)
        self.assertEqual(rig.cost_usd(-5, 10), 0)

    def test_builtin_rates_cover_the_stage_2b_node(self):
        self.assertIn("p4de.24xlarge", rig.RATES)


class TestRealAwsShapes(unittest.TestCase):
    """Bugs the first real AWS launch (2026-10-13) found, which the fake aws could not."""

    # the keys the real RunInstances MetadataOptions accepts (from the CLI's own validation message)
    METADATA_KEYS = {"HttpTokens", "HttpPutResponseHopLimit", "HttpEndpoint", "HttpProtocolIpv6", "InstanceMetadataTags"}

    def test_run_instances_metadata_options_use_keys_the_real_cli_accepts(self):
        src = (ROOT / "tools" / "rig" / "rig.py").read_text()
        m = re.search(r'"--metadata-options", "([^"]+)"', src)
        self.assertTrue(m)
        keys = {kv.split("=")[0] for kv in m.group(1).split(",")}
        self.assertLessEqual(keys, self.METADATA_KEYS)
        self.assertIn("HttpTokens=required", m.group(1))

    def test_a_multi_line_cli_error_keeps_its_cause(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = pathlib.Path(tmp) / "aws"
            fake.write_text("#!/bin/sh\necho 'aws: [ERROR]: An error occurred (ParamValidation): Parameter validation failed:' >&2\n"
                            "echo 'Unknown parameter in MetadataOptions: \"Bogus\", must be one of: HttpTokens' >&2\nexit 252\n")
            fake.chmod(0o755)
            with mock.patch.dict(os.environ, {"PW_AWS": str(fake)}):
                with self.assertRaises(rig.AwsError) as cm:
                    rig.aws(["ec2", "run-instances"], "us-east-1")
        self.assertEqual(cm.exception.code, "ParamValidation")
        self.assertIn("Unknown parameter in MetadataOptions", str(cm.exception))


class TestHomeSafety(RigCase):
    def test_a_scratchpad_is_refused(self):
        os.environ.pop("PW_RIGS_ALLOW_UNSAFE_HOME")
        for bad in ["/tmp/claude-1002/x/scratchpad/rigs", "/home/u/scratchpad/k", "/tmp/claude-5/whatever"]:
            with self.assertRaises(rig.RigError, msg=bad):
                rig.check_home(bad)

    def test_launch_in_a_scratchpad_fails_before_any_aws_call(self):
        os.environ.pop("PW_RIGS_ALLOW_UNSAFE_HOME")
        os.environ["PW_RIGS_HOME"] = str(self.tmp / "scratchpad" / "rigs")
        rc, _, err = self.launch()
        self.assertEqual(rc, 2)
        self.assertIn("scratchpad", err)
        self.assertEqual(self.calls(), [])

    def test_default_home_is_under_the_users_cache(self):
        os.environ.pop("PW_RIGS_HOME")
        os.environ.pop("PW_RIGS_ALLOW_UNSAFE_HOME")
        self.assertEqual(rig.home(), (self.tmp / "home" / ".cache" / "pw-rigs").resolve())


class TestLaunch(RigCase):
    def test_launch_creates_a_private_key_outside_any_scratchpad(self):
        rc, out, err = self.launch()
        self.assertEqual(rc, 0, err)
        key = self.tmp / "rigs" / "pw-t1" / "key.pem"
        self.assertTrue(key.is_file())
        self.assertEqual(stat.S_IMODE(key.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(key.parent.stat().st_mode), 0o700)
        self.assertNotIn("scratchpad", str(key))
        self.assertIn(f"ssh -i {key} ", out)
        self.assertIn("ubuntu@203.0.113.10", out)
        self.assertIn("BEGIN OPENSSH PRIVATE KEY", key.read_text())

    def test_instance_is_configured_to_terminate_itself(self):
        self.launch()
        (call,) = self.called("run-instances")
        self.assertEqual(call[call.index("--instance-initiated-shutdown-behavior") + 1], "terminate")
        self.assertEqual(call[call.index("--metadata-options") + 1].split(",")[0], "HttpTokens=required")
        inst = next(iter(self.state()["instances"].values()))
        self.assertEqual(inst["ShutdownBehavior"], "terminate")
        bdm = inst["Ebs"]
        self.assertEqual((bdm["VolumeSize"], bdm["VolumeType"], bdm["DeleteOnTermination"]), (100, "gp3", True))

    def test_security_group_allows_ssh_from_the_callers_address_only(self):
        self.launch()
        (sg,) = self.state()["sgs"].values()
        self.assertEqual(sg["ingress"], [{"port": "22", "cidr": "198.51.100.7/32"}])
        self.assertEqual(len(self.called("authorize-security-group-ingress")), 1)

    def test_deadline_is_absolute_and_in_the_user_data_and_tags(self):
        rc, out, err = self.launch(rate="10", hours="8", dollars="1000", disk="100")
        self.assertEqual(rc, 0, err)
        inst = next(iter(self.state()["instances"].values()))
        want = int(NOW + 8 * 3600)
        self.assertIn(f"printf '%s\\n' {want} > /etc/pw-rig/deadline", inst["UserData"])
        tags = {t["Key"]: t["Value"] for t in inst["Tags"]}
        self.assertEqual(tags["pw-deadline"], str(want))
        self.assertEqual(tags["Name"], "pw-t1")
        self.assertEqual(tags["pw-rig-tool"], "1")
        r = json.loads((self.tmp / "rigs" / "pw-t1" / "rig.json").read_text())
        self.assertEqual(r["deadline"], want)
        self.assertIn("timer verified", out)

    def test_dollar_cap_shortens_the_life(self):
        # $30 at $10/h plus the disk is a bit under 3 hours, not the 8 asked for
        rc, out, err = self.launch(rate="10", hours="8", dollars="30", disk="100")
        self.assertEqual(rc, 0, err)
        r = json.loads((self.tmp / "rigs" / "pw-t1" / "rig.json").read_text())
        hourly = rig.hourly_cost(10, 100)
        self.assertAlmostEqual(r["deadline"], NOW + 30 / hourly * 3600, delta=1)
        self.assertLess(r["deadline"], NOW + 3 * 3600)
        self.assertIn("max-dollars", err)

    def test_the_timer_is_made_tamper_resistant_and_fails_closed(self):
        ud = rig.user_data("pw-x", NOW + 100)
        self.assertIn("chattr +i /etc/pw-rig/deadline", ud)
        self.assertIn("chown root:root /etc/pw-rig/deadline", ud)
        self.assertIn("systemctl enable --now pw-rig-watchdog.timer", ud)
        self.assertIn("shutdown -h now", ud)
        self.assertIn("missing or unreadable: terminating", ud)
        self.assertNotIn("shutdown -c", ud)

    def test_a_timer_that_cannot_be_verified_is_loud(self):
        os.environ["FAKE_SSH_DOWN"] = "1"
        rc, out, err = self.run_rig("launch", "--tag", "t2", "--instance-type", "p4de.24xlarge", "--rate", "10", "--max-hours", "1",
                                    "--max-dollars", "100", "--ssh-wait", "0.2")
        self.assertEqual(rc, 3)
        self.assertIn("TIMER NOT VERIFIED", err)
        self.assertTrue(any("timer-unverified" in l for l in self.ledger_lines()))

    def test_ledger_records_the_launch(self):
        self.launch()
        lines = self.ledger_lines()
        self.assertEqual(lines[0], "time\ttag\tevent\tdetails")
        self.assertIn("\tpw-t1\tlaunch\t", lines[1])
        self.assertIn("max_dollars=1000", lines[1])

    def test_newest_matching_ami_is_used(self):
        self.launch()
        (call,) = self.called("run-instances")
        self.assertEqual(call[call.index("--image-id") + 1], "ami-new")

    def test_a_tag_that_exists_locally_is_not_reused_and_its_key_survives(self):
        self.launch()
        key = self.tmp / "rigs" / "pw-t1" / "key.pem"
        before = key.read_text()
        rc, _, err = self.launch()
        self.assertEqual(rc, 2)
        self.assertIn("refusing to overwrite its key", err)
        self.assertEqual(key.read_text(), before)
        self.assertEqual(len(self.called("create-key-pair")), 1)

    def test_tag_gets_the_pw_prefix_and_is_validated(self):
        rc, _, err = self.run_rig("launch", "--tag", "Bad Tag!", "--instance-type", "g5.xlarge", "--max-hours", "1", "--max-dollars", "5")
        self.assertEqual(rc, 2)
        self.assertEqual(rig.norm_tag("stage2b"), "pw-stage2b")

    def test_unknown_instance_type_needs_a_rate(self):
        rc, _, err = self.run_rig("launch", "--tag", "t", "--instance-type", "zz9.huge", "--max-hours", "1", "--max-dollars", "5")
        self.assertEqual(rc, 2)
        self.assertIn("--rate", err)
        self.assertEqual(self.calls(), [])

    def test_dry_run_makes_no_aws_call_and_creates_nothing(self):
        rc, out, _ = self.run_rig("launch", "--tag", "t", "--instance-type", "p4de.24xlarge", "--max-hours", "2", "--max-dollars", "200", "--dry-run")
        self.assertEqual(rc, 0)
        self.assertIn("DRY RUN", out)
        self.assertIn("/etc/pw-rig/deadline", out)
        self.assertEqual(self.calls(), [])
        self.assertFalse((self.tmp / "rigs" / "pw-t").exists())


class TestAzFallback(RigCase):
    def test_falls_through_to_the_next_az_on_insufficient_capacity(self):
        self.set_state(capacity_fail_azs=["us-east-1a", "us-east-1b"])
        rc, out, err = self.launch()
        self.assertEqual(rc, 0, err)
        runs = self.called("run-instances")
        self.assertEqual([c[c.index("--subnet-id") + 1] for c in runs], ["subnet-a", "subnet-b", "subnet-c"])
        r = json.loads((self.tmp / "rigs" / "pw-t1" / "rig.json").read_text())
        self.assertEqual(r["az"], "us-east-1c")
        self.assertIn("InsufficientInstanceCapacity", err)
        self.assertEqual(len(self.called("create-security-group")), 1)   # one group serves every attempt

    def test_explicit_az_list_is_honoured_in_order(self):
        self.set_state(capacity_fail_azs=["us-east-1d"])
        rc, _, err = self.launch(extra=("--azs", "us-east-1d,us-east-1f"))
        self.assertEqual(rc, 0, err)
        self.assertEqual(json.loads((self.tmp / "rigs" / "pw-t1" / "rig.json").read_text())["az"], "us-east-1f")

    def test_deadline_is_recomputed_per_attempt(self):
        self.set_state(capacity_fail_azs=["us-east-1a"])
        self.launch()
        inst = next(iter(self.state()["instances"].values()))
        self.assertIn(str(int(NOW + 8 * 3600)), inst["UserData"])

    def test_no_capacity_anywhere_releases_the_key_group_and_directory(self):
        self.set_state(capacity_fail_azs=[f"us-east-1{z}" for z in "abcdf"])
        rc, _, err = self.launch()
        self.assertEqual(rc, 2)
        self.assertIn("no capacity", err)
        s = self.state()
        self.assertEqual((s["keys"], s["sgs"], s["instances"]), ({}, {}, {}))
        self.assertFalse((self.tmp / "rigs" / "pw-t1").exists())
        self.assertEqual(len(self.called("run-instances")), 5)

    def test_other_errors_do_not_fall_through_and_clean_up(self):
        self.set_state(run_error="VcpuLimitExceeded")
        rc, _, err = self.launch()
        self.assertEqual(rc, 2)
        self.assertIn("VcpuLimitExceeded", err)
        self.assertEqual(len(self.called("run-instances")), 1)
        s = self.state()
        self.assertEqual((s["keys"], s["sgs"]), ({}, {}))
        self.assertFalse((self.tmp / "rigs" / "pw-t1").exists())


class TestTeardown(RigCase):
    def test_teardown_terminates_verifies_and_logs_a_cost(self):
        self.launch()
        hourly = rig.hourly_cost(10, 100)
        os.environ["PW_RIG_NOW"] = str(NOW + 5400)           # 1.5 h later
        self.set_state(now=NOW + 5400)
        rc, out, err = self.run_rig("teardown", "pw-t1")
        self.assertEqual(rc, 0, err)
        s = self.state()
        self.assertEqual({i["State"]["Name"] for i in s["instances"].values()}, {"terminated"})
        self.assertEqual((s["volumes"], s["keys"], s["sgs"]), ({}, {}, {}))
        self.assertFalse((self.tmp / "rigs" / "pw-t1" / "key.pem").exists())   # the private key is gone locally too
        last = self.ledger_lines()[-1]
        self.assertIn("\tteardown\t", last)
        self.assertIn(f"usd={1.5 * hourly:.2f}", last)
        self.assertIn("hours=1.50", last)
        self.assertIn("estimated", out)

    def test_teardown_is_idempotent(self):
        self.launch()
        self.assertEqual(self.run_rig("teardown", "pw-t1")[0], 0)
        n = len(self.ledger_lines())
        rc, out, err = self.run_rig("teardown", "pw-t1")
        self.assertEqual(rc, 0, err)
        self.assertEqual(len(self.ledger_lines()), n)             # no second cost line
        self.assertIn("already torn down", out)
        self.assertEqual(self.run_rig("teardown", "pw-never-existed")[0], 0)

    def test_leaked_volume_is_deleted_and_reported(self):
        self.launch()
        self.set_state(leak_volume=True)
        rc, _, err = self.run_rig("teardown", "pw-t1", "--wait", "0")
        self.assertEqual(rc, 0, err)
        self.assertEqual(len(self.called("delete-volume")), 1)
        self.assertEqual(self.state()["volumes"], {})

    def test_security_group_dependency_violation_is_retried(self):
        self.launch()
        s = self.state()
        for g in s["sgs"].values():
            g["busy"] = 3
        self.set_state(sgs=s["sgs"])
        rc, _, err = self.run_rig("teardown", "pw-t1")
        self.assertEqual(rc, 0, err)
        self.assertEqual(len(self.called("delete-security-group")), 4)
        self.assertEqual(self.state()["sgs"], {})

    def test_a_group_that_never_frees_is_a_failure_not_a_silent_success(self):
        self.launch()
        s = self.state()
        for g in s["sgs"].values():
            g["busy"] = 1000
        self.set_state(sgs=s["sgs"])
        rc, _, err = self.run_rig("teardown", "pw-t1")
        self.assertEqual(rc, 1)
        self.assertIn("PROBLEM", err)

    def test_an_instance_not_launched_by_the_tool_is_not_terminated(self):
        self.set_state(extra_instances=[{"InstanceId": "i-foreign", "InstanceType": "g5.xlarge", "State": {"Name": "running"},
                                         "Tags": [{"Key": "Name", "Value": "pw-theirs"}], "LaunchTime": "2027-01-15T08:00:00.000Z"}])
        rc, _, err = self.run_rig("teardown", "pw-theirs")
        self.assertEqual(rc, 2)
        self.assertIn("may belong to someone else", err)
        self.assertEqual(self.called("terminate-instances"), [])


class TestStatus(RigCase):
    def at(self, seconds_after_launch):
        os.environ["PW_RIG_NOW"] = str(NOW + seconds_after_launch)

    def test_nothing_running_is_ok(self):
        rc, out, _ = self.run_rig("status")
        self.assertEqual(rc, 0)
        self.assertIn("ok", out)

    def test_a_rig_inside_its_cap_and_deadline_is_ok(self):
        self.launch(hours="8", dollars="1000")
        self.at(3600)
        rc, out, _ = self.run_rig("status")
        self.assertEqual(rc, 0, out)
        self.assertIn("pw-t1", out)
        self.assertIn("running", out)

    def test_past_the_deadline_is_flagged(self):
        self.launch(hours="2", dollars="1000")
        self.at(2 * 3600 + 300)                                # inside the grace period
        self.assertEqual(self.run_rig("status")[0], 0)
        self.at(2 * 3600 + rig.DEADLINE_GRACE_S + 60)
        rc, out, _ = self.run_rig("status")
        self.assertEqual(rc, 1)
        self.assertIn("PAST-DEADLINE", out)
        self.assertNotIn("OVER-CAP", out)

    def test_over_the_dollar_cap_is_flagged(self):
        launch = "2027-01-15T07:00:00.000Z"
        t = rig._parse_time(launch)
        os.environ["PW_RIG_NOW"] = str(t + 4 * 3600)
        self.set_state(extra_instances=[{"InstanceId": "i-1", "InstanceType": "g5.xlarge", "State": {"Name": "running"}, "LaunchTime": launch,
                                         "Tags": [{"Key": "Name", "Value": "pw-over"}, {"Key": "pw-rig-tool", "Value": "1"},
                                                  {"Key": "pw-max-dollars", "Value": "5"}, {"Key": "pw-hourly-usd", "Value": "2"},
                                                  {"Key": "pw-deadline", "Value": str(int(t + 10 * 3600))}]}])
        rc, out, _ = self.run_rig("status")
        self.assertEqual(rc, 1)
        self.assertIn("OVER-CAP", out)
        self.assertNotIn("PAST-DEADLINE", out)
        row = next(l for l in out.splitlines() if l.startswith("pw-over"))
        self.assertIn("8.00", row)                              # 4 h x $2/h running cost

    def test_under_the_dollar_cap_is_not_flagged(self):
        launch = "2027-01-15T07:00:00.000Z"
        t = rig._parse_time(launch)
        os.environ["PW_RIG_NOW"] = str(t + 2 * 3600)
        self.set_state(extra_instances=[{"InstanceId": "i-1", "InstanceType": "g5.xlarge", "State": {"Name": "running"}, "LaunchTime": launch,
                                         "Tags": [{"Key": "Name", "Value": "pw-fine"}, {"Key": "pw-rig-tool", "Value": "1"},
                                                  {"Key": "pw-max-dollars", "Value": "5"}, {"Key": "pw-hourly-usd", "Value": "2"},
                                                  {"Key": "pw-deadline", "Value": str(int(t + 10 * 3600))}]}])
        self.assertEqual(self.run_rig("status")[0], 0)

    def test_an_instance_with_no_cap_anywhere_is_flagged(self):
        self.set_state(extra_instances=[{"InstanceId": "i-2", "InstanceType": "g5.xlarge", "State": {"Name": "running"},
                                         "LaunchTime": "2027-01-15T07:00:00.000Z", "Tags": [{"Key": "Name", "Value": "pw-nocap"}]}])
        rc, out, _ = self.run_rig("status")
        self.assertEqual(rc, 1)
        self.assertIn("NO-CAP-RECORD", out)

    def test_a_leftover_volume_is_flagged(self):
        self.launch()
        self.set_state(leak_volume=True)
        s = self.state()
        for i in s["instances"].values():
            i["State"] = {"Name": "terminated"}
        for v in s["volumes"].values():
            v["State"] = "available"
        self.set_state(instances=s["instances"], volumes=s["volumes"])
        self.at(60)
        rc, out, _ = self.run_rig("status")
        self.assertEqual(rc, 1)
        self.assertIn("ORPHAN-VOLUME", out)

    def test_local_record_without_an_instance_is_noted(self):
        self.launch()
        s = self.state()
        s["instances"] = {}
        self.set_state(instances={})
        rc, out, _ = self.run_rig("status")
        self.assertIn("run teardown.sh pw-t1", out)


class TestExtend(RigCase):
    def setUp(self):
        super().setUp()
        self.assertEqual(self.launch(hours="2", dollars="1000")[0], 0)
        self.old = int(NOW + 2 * 3600)

    def rig_json(self):
        return json.loads((self.tmp / "rigs" / "pw-t1" / "rig.json").read_text())

    def test_extension_moves_the_deadline_on_the_box_and_logs_the_reason(self):
        rc, out, err = self.run_rig("extend", "pw-t1", "3", "--reason", "weights still downloading for deepseek-v3.2")
        self.assertEqual(rc, 0, err)
        new = self.old + 3 * 3600
        self.assertEqual((self.tmp / "box" / "deadline").read_text().strip(), str(new))
        self.assertEqual(self.rig_json()["deadline"], new)
        last = self.ledger_lines()[-1]
        self.assertIn("\textend\t", last)
        self.assertIn("reason=weights still downloading for deepseek-v3.2", last)
        self.assertIn("hours=3.0", last)
        tags = {t["Key"]: t["Value"] for t in next(iter(self.state()["instances"].values()))["Tags"]}
        self.assertEqual(tags["pw-deadline"], str(new))
        ssh_cmds = [json.loads(l)["cmd"] for l in (self.tmp / "ssh.log").read_text().splitlines()]
        self.assertTrue(any("chattr -i /etc/pw-rig/deadline" in c and "chattr +i /etc/pw-rig/deadline" in c for c in ssh_cmds))

    def test_a_reason_is_mandatory(self):
        with self.assertRaises(SystemExit):
            self.run_rig("extend", "pw-t1", "1")
        rc, _, err = self.run_rig("extend", "pw-t1", "1", "--reason", "x")
        self.assertEqual(rc, 2)
        self.assertEqual(self.rig_json()["deadline"], self.old)

    def test_extension_over_the_dollar_cap_is_refused_unless_raised_explicitly(self):
        # cap $30 at ~$10.1/h allows about 2.9 h; asking for 2 more hours on top of the first ~2.9 is over
        self.run_rig("teardown", "pw-t1")
        (self.tmp / "box" / "deadline").unlink()
        self.assertEqual(self.launch(tag="pw-t3", hours="8", dollars="30")[0], 0)
        rc, _, err = self.run_rig("extend", "pw-t3", "2", "--reason", "needs more time to finish")
        self.assertEqual(rc, 2)
        self.assertIn("over the cap", err)
        r = json.loads((self.tmp / "rigs" / "pw-t3" / "rig.json").read_text())
        self.assertLess(r["deadline"], NOW + 3 * 3600)      # refused: still the capped original
        rc, _, err = self.run_rig("extend", "pw-t3", "2", "--reason", "needs more time to finish", "--max-dollars", "100")
        self.assertEqual(rc, 0, err)
        self.assertEqual(json.loads((self.tmp / "rigs" / "pw-t3" / "rig.json").read_text())["max_dollars"], 100)
        self.assertIn("cap_usd=100", self.ledger_lines()[-1])

    def test_a_passed_deadline_cannot_be_extended(self):
        os.environ["PW_RIG_NOW"] = str(self.old + 10)
        rc, _, err = self.run_rig("extend", "pw-t1", "1", "--reason", "too late, should be refused")
        self.assertEqual(rc, 2)
        self.assertIn("has passed", err)

    def test_unknown_or_closed_rigs_are_refused(self):
        self.assertEqual(self.run_rig("extend", "pw-nope", "1", "--reason", "does not exist at all")[0], 2)
        self.run_rig("teardown", "pw-t1")
        self.assertEqual(self.run_rig("extend", "pw-t1", "1", "--reason", "already torn down rig")[0], 2)

    def test_a_failed_ssh_changes_nothing(self):
        os.environ["FAKE_SSH_DOWN"] = "1"
        n = len(self.ledger_lines())
        rc, _, err = self.run_rig("extend", "pw-t1", "1", "--reason", "ssh is down so this fails")
        self.assertEqual(rc, 2)
        self.assertEqual(self.rig_json()["deadline"], self.old)
        self.assertEqual(len(self.ledger_lines()), n)


if __name__ == "__main__":
    unittest.main()
