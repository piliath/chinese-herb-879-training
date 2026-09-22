# -*- coding: utf-8 -*-
"""
AutoDL SSH 助手 (paramiko)
用法:
  python ssh_helper.py run "<shell命令>"              # 执行远程命令并打印输出
  python ssh_helper.py upload <本地文件> <远程路径>     # SFTP 上传
  python ssh_helper.py download <远程文件> <本地路径>   # SFTP 下载
  python ssh_helper.py mkdir <远程目录>               # 创建远程目录(含父目录)
连接参数通过环境变量覆盖: SSH_HOST / SSH_PORT / SSH_USER / SSH_PASS
"""
import os
import sys
import paramiko

HOST = os.environ.get('SSH_HOST', 'connect.nmb2.seetacloud.com')
PORT = int(os.environ.get('SSH_PORT', '11277'))
USER = os.environ.get('SSH_USER', 'root')
PASS = os.environ.get('SSH_PASS', 'UF3CxNwoo75x')


def connect():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=30)
    return client


def run(cmd, timeout=1800):
    client = connect()
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode('utf-8', errors='replace')
    err = stderr.read().decode('utf-8', errors='replace')
    code = stdout.channel.recv_exit_status()
    client.close()
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    if out:
        sys.stdout.write(out if out.endswith('\n') else out + '\n')
    if err:
        sys.stdout.write('[stderr] ' + err if err.endswith('\n') else '[stderr] ' + err + '\n')
    print(f'[exit] {code}')
    return code


def upload(local, remote):
    # paramiko 5.0 put/putfo 在该服务器上二次 open 报 ENOENT, 改用手动写字节
    with open(local, 'rb') as lf:
        data = lf.read()
    client = connect()
    sftp = client.open_sftp()
    with sftp.open(remote, 'wb') as rf:
        rf.write(data)
    sftp.close()
    client.close()
    print(f'[upload] {local} -> {remote} ({len(data)/1024/1024:.1f} MB)')


def download(remote, local):
    client = connect()
    sftp = client.open_sftp()
    sftp.get(remote, local)
    sftp.close()
    client.close()
    print(f'[download] {remote} -> {local}')


def mkdir(remote):
    client = connect()
    sftp = client.open_sftp()
    parts = remote.strip('/').split('/')
    cur = ''
    for p in parts:
        cur += '/' + p
        try:
            sftp.stat(cur)
        except FileNotFoundError:
            sftp.mkdir(cur)
    sftp.close()
    client.close()
    print(f'[mkdir] {remote}')


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    op = sys.argv[1]
    if op == 'run':
        sys.exit(run(sys.argv[2]))
    elif op == 'upload':
        upload(sys.argv[2], sys.argv[3])
    elif op == 'download':
        download(sys.argv[2], sys.argv[3])
    elif op == 'mkdir':
        mkdir(sys.argv[2])
    else:
        print('未知操作')
        sys.exit(1)


if __name__ == '__main__':
    main()
