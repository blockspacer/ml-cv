# conding: utf-8


# paramiko

# windows ssh http://www.freesshd.com/, putty
# http://www.techrepublic.com/blog/tr-dojo/set-up-a-free-ssh-server-on-windows-7-with-freesshd/
# http://www.janaka.co.uk/2012/12/how-to-configure-freesshd-on-windows.html
#
# Win:
# https://www.youtube.com/watch?v=Zei2DPCnMl4
# https://askubuntu.com/questions/204400/ssh-public-key-no-supported-authentication-methods-available-server-sent-publ
#
# Not working:
# https://wthwdik.wordpress.com/2011/03/28/how-to-get-freesshd-public-key-authentication-to-work/
# на винде так и не заработало

import paramiko
import select
import socket
# from socket import Queue
import sys
# import Queue  # no module

if __name__ == '__main__':
    # https://stackoverflow.com/questions/760978/long-running-ssh-commands-in-python-paramiko-module-and-how-to-end-them
    if 0:
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.connect('localhost')
        channel = client.get_transport().open_session()
        channel.exec_command("tail -f /var/log/everything/current")
        while True:
            if channel.exit_status_ready():
                break
            rl, wl, xl = select.select([channel], [], [], 0.0)
            if len(rl) > 0:
                # print
                channel.recv(1024)

    if 1:
        # https://pymotw.com/2/select/
        # no poll() on Windows
        pass
