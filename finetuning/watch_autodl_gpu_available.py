#!/usr/bin/env python3
import argparse
import json
import os
import smtplib
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from email.mime.text import MIMEText
from email.utils import formataddr


API_URL = "https://api.autodl.com/api/v1/machine/search"


def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Watch AutoDL GPU availability and send email when a matching GPU becomes available."
    )
    parser.add_argument(
        "--gpu-name",
        default="RTX PRO 6000",
        help="Exact AutoDL GPU type name, for example 'RTX PRO 6000'.",
    )
    parser.add_argument(
        "--region-sign",
        default="",
        help="Optional AutoDL region sign filter. Empty means all regions.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=60,
        help="Polling interval in seconds.",
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
        default=os.environ.get("SMTP_SENDER_NAME", "AutoDL GPU Watcher"),
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
        help="QQ SMTP auth code.",
    )
    parser.add_argument(
        "--autodl-token",
        default=os.environ.get("AUTODL_TOKEN", ""),
        help="AutoDL localStorage token used in Authorization header.",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=20,
        help="Number of machines to ask from the API per check.",
    )
    parser.add_argument(
        "--notify-on-every-hit",
        action="store_true",
        help="Send an email every time a positive result is seen. By default only the first hit is emailed.",
    )
    return parser.parse_args()


def build_payload(args):
    return {
        "charge_type": "payg",
        "region_sign": args.region_sign,
        "gpu_type_name": [args.gpu_name],
        "machine_tag_name": [],
        "gpu_idle_num": 1,
        "mount_net_disk": False,
        "instance_disk_size_order": "",
        "date_range": "",
        "date_from": "",
        "date_to": "",
        "page_index": 1,
        "page_size": args.page_size,
        "pay_price_order": "",
        "gpu_idle_type": "",
        "default_order": True,
        "region_sign_list": [args.region_sign] if args.region_sign else [],
    }


def post_json(url, payload, token):
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": token,
        "AppVersion": "0.0.0",
        "User-Agent": "autodl-gpu-watcher/1.0",
    }
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=30) as response:
        body = response.read().decode("utf-8")
        return json.loads(body)


def send_email(args, subject, body):
    if not args.smtp_auth_code:
        raise RuntimeError("Missing SMTP auth code.")

    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = formataddr((args.sender_name, args.sender_email))
    msg["To"] = args.recipient

    with smtplib.SMTP_SSL(args.smtp_host, args.smtp_port, timeout=30) as server:
        server.login(args.sender_email, args.smtp_auth_code)
        server.sendmail(args.sender_email, [args.recipient], msg.as_string())


def summarize_machine(machine):
    region = machine.get("region_name") or machine.get("region_sign") or "unknown-region"
    gpu_name = machine.get("gpu_name") or machine.get("gpu_type_name") or "unknown-gpu"
    idle = machine.get("gpu_idle_num", "?")
    price = machine.get("payg_price") or machine.get("current_price") or machine.get("price") or "?"
    return f"region={region}, gpu={gpu_name}, idle={idle}, price={price}"


def main():
    args = parse_args()
    if not args.autodl_token:
        print("[gpu-watch] missing AUTODL_TOKEN", file=sys.stderr)
        return 1

    payload = build_payload(args)
    print(f"[{now_str()}] watching AutoDL gpu='{args.gpu_name}' region='{args.region_sign or 'ALL'}'", flush=True)

    notified = False
    while True:
        try:
            response = post_json(API_URL, payload, args.autodl_token)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            print(f"[{now_str()}] HTTP error {exc.code}: {body}", flush=True)
            time.sleep(args.poll_seconds)
            continue
        except Exception as exc:
            print(f"[{now_str()}] request failed: {exc}", flush=True)
            time.sleep(args.poll_seconds)
            continue

        code = response.get("code")
        if code != "Success":
            print(f"[{now_str()}] API returned code={code} msg={response.get('msg', '')}", flush=True)
            time.sleep(args.poll_seconds)
            continue

        data = response.get("data") or {}
        total = data.get("result_total", 0)
        machines = data.get("list") or []
        print(f"[{now_str()}] result_total={total}", flush=True)

        if total and (args.notify_on_every_hit or not notified):
            summary_lines = [summarize_machine(machine) for machine in machines[:10]]
            subject = f"AutoDL 有空闲 GPU 了: {args.gpu_name}"
            body = (
                f"检测时间: {now_str()}\n"
                f"GPU 型号: {args.gpu_name}\n"
                f"区域过滤: {args.region_sign or 'ALL'}\n"
                f"匹配机器数: {total}\n\n"
                f"前几台机器:\n" + "\n".join(summary_lines)
            )
            send_email(args, subject, body)
            notified = True
            print(f"[{now_str()}] availability email sent to {args.recipient}", flush=True)

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    sys.exit(main())
