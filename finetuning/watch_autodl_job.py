#!/usr/bin/env python3
import argparse
import os
import shlex
import shutil
import smtplib
import subprocess
import sys
import time
from datetime import datetime
from email.mime.text import MIMEText
from email.utils import formataddr


DEFAULT_ASKPASS_SCRIPT = "/media/data/zhangjingyi/ImAge/finetuning/ssh_askpass_autodl.sh"


def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Watch a remote AutoDL job over SSH and email when it finishes."
    )
    parser.add_argument("--host", required=True, help="SSH host, for example connect.bjb1.seetacloud.com")
    parser.add_argument("--port", type=int, default=22, help="SSH port")
    parser.add_argument("--user", default="root", help="SSH username")
    parser.add_argument(
        "--process-keyword",
        default="train_mixed_resume.py",
        help="Keyword used to match the remote process in ps output.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=30,
        help="Polling interval in seconds.",
    )
    parser.add_argument(
        "--startup-timeout-minutes",
        type=int,
        default=0,
        help="Optional timeout for waiting until the job is first seen running. 0 means wait forever.",
    )
    parser.add_argument(
        "--recipient",
        default="2327667951@qq.com",
        help="Notification recipient email address.",
    )
    parser.add_argument(
        "--sender-email",
        default=os.environ.get("SMTP_SENDER_EMAIL", "2327667951@qq.com"),
        help="SMTP sender email address.",
    )
    parser.add_argument(
        "--sender-name",
        default=os.environ.get("SMTP_SENDER_NAME", "AutoDL Watcher"),
        help="Display name used in the sender field.",
    )
    parser.add_argument(
        "--smtp-host",
        default=os.environ.get("SMTP_HOST", "smtp.qq.com"),
        help="SMTP server host.",
    )
    parser.add_argument(
        "--smtp-port",
        type=int,
        default=int(os.environ.get("SMTP_PORT", "465")),
        help="SMTP SSL port.",
    )
    parser.add_argument(
        "--smtp-auth-code",
        default=os.environ.get("SMTP_AUTH_CODE", ""),
        help="SMTP auth code. For QQ mail this is the SMTP authorization code, not the mailbox password.",
    )
    parser.add_argument(
        "--ssh-password",
        default=os.environ.get("AUTODL_SSH_PASSWORD", ""),
        help="SSH password. If omitted, normal ssh key auth is used.",
    )
    parser.add_argument(
        "--remote-check-command",
        default="",
        help="Optional full remote shell command. If provided, it overrides --process-keyword logic.",
    )
    parser.add_argument(
        "--ssh-askpass-script",
        default=os.environ.get("SSH_ASKPASS_SCRIPT", DEFAULT_ASKPASS_SCRIPT),
        help="Askpass helper script used when --ssh-password is provided and sshpass is unavailable.",
    )
    return parser.parse_args()


def build_remote_command(args):
    if args.remote_check_command:
        return args.remote_check_command

    keyword = shlex.quote(args.process_keyword)
    return (
        "ps -ef | grep -F {kw} | grep -v grep || true".format(kw=keyword)
    )


def run_ssh(args, remote_command):
    ssh_cmd = [
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "ConnectTimeout=10",
        "-p",
        str(args.port),
        f"{args.user}@{args.host}",
        remote_command,
    ]

    if args.ssh_password:
        if shutil.which("sshpass") is not None:
            ssh_cmd = ["sshpass", "-p", args.ssh_password] + ssh_cmd
            result = subprocess.run(
                ssh_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            return result

        askpass_script = args.ssh_askpass_script
        if not os.path.exists(askpass_script):
            raise RuntimeError(
                "sshpass is not installed and askpass script is missing."
            )

        env = os.environ.copy()
        env["AUTODL_SSH_PASSWORD"] = args.ssh_password
        env["SSH_ASKPASS"] = askpass_script
        env["SSH_ASKPASS_REQUIRE"] = "force"
        env["DISPLAY"] = env.get("DISPLAY", "dummy:0")

        result = subprocess.run(
            ["setsid"] + ssh_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            stdin=subprocess.DEVNULL,
            env=env,
        )
        return result

    result = subprocess.run(
        ssh_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result


def send_email(args, subject, body):
    if not args.smtp_auth_code:
        raise RuntimeError(
            "Missing SMTP auth code. Set SMTP_AUTH_CODE or pass --smtp-auth-code."
        )

    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = formataddr((args.sender_name, args.sender_email))
    msg["To"] = args.recipient

    with smtplib.SMTP_SSL(args.smtp_host, args.smtp_port, timeout=30) as server:
        server.login(args.sender_email, args.smtp_auth_code)
        server.sendmail(args.sender_email, [args.recipient], msg.as_string())


def main():
    args = parse_args()
    remote_command = build_remote_command(args)
    print(f"[{now_str()}] watching {args.user}@{args.host}:{args.port}", flush=True)
    print(f"[{now_str()}] remote check command: {remote_command}", flush=True)

    seen_running = False
    start_time = time.time()

    while True:
        try:
            result = run_ssh(args, remote_command)
        except Exception as exc:
            print(f"[{now_str()}] SSH check failed: {exc}", flush=True)
            time.sleep(args.poll_seconds)
            continue

        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

        if result.returncode != 0:
            print(
                f"[{now_str()}] server not ready or SSH failed: {stderr or 'unknown error'}",
                flush=True,
            )
            time.sleep(args.poll_seconds)
            continue

        is_running = bool(stdout)

        if is_running:
            if not seen_running:
                print(f"[{now_str()}] target job detected as running.", flush=True)
            else:
                print(f"[{now_str()}] target job still running.", flush=True)
            seen_running = True
        else:
            if seen_running:
                subject = f"AutoDL 任务已结束: {args.process_keyword}"
                body = (
                    f"检测时间: {now_str()}\n"
                    f"机器: {args.user}@{args.host}:{args.port}\n"
                    f"监控关键字: {args.process_keyword}\n"
                    f"结果: 任务已结束，远程进程已消失。\n"
                )
                send_email(args, subject, body)
                print(f"[{now_str()}] finish email sent to {args.recipient}", flush=True)
                return

            waited_minutes = (time.time() - start_time) / 60.0
            print(f"[{now_str()}] target job not running yet.", flush=True)
            if (
                args.startup_timeout_minutes > 0
                and waited_minutes >= args.startup_timeout_minutes
            ):
                print(
                    f"[{now_str()}] startup timeout reached before the job ever appeared.",
                    flush=True,
                )
                return 1

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    sys.exit(main())
