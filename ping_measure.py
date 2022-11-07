################################################
#            Generate sample pings
################################################

import datetime
import os

import pandas
from datetime import datetime
from scapy.all import sr, IP, TCP
from scapy.layers.inet import traceroute, ICMP


def measure(ip, traceroute_attempts=32, pings=int(1e3),
            savedir="./diagnostics", name_prefix="dia_lan", return_type="file", timeout=1):
    """
    measures ping times for ip. First traceroute_attempts traceroutes are performed, then the ip is
    pinged. Each ping is then saved into a dataframe, which is the return value
    """
    print(f"measuring ping time for IP: {ip}")

    traceroute_counter = {}
    for i in range(traceroute_attempts):
        res, unans = traceroute(ip, verbose=False)
        trace = res.get_trace()
        for addr in trace:
            traceroute_counter[addr] = traceroute_counter.get(addr, 0) + len(trace[addr])
    avg_traceroute={}
    for cnt in traceroute_counter:
        avg_traceroute[cnt] = traceroute_counter[cnt] / traceroute_attempts

    print("avg traceroutes: " + str(avg_traceroute))

    pre_df_array = []

    for i in range(pings):
        print(f"perform ping {i}/{pings}")
        ans, unans = sr(IP(dst=ip) / ICMP(), verbose=False, timeout=timeout)
        for src, resp in ans:
            if IP in resp and resp[IP].src == ip:
                time = resp.time - src.sent_time
                print(time)
                # [ip, avg_traceroute, ping time]
                to_be_added = [src.sprintf("%IP.dst%"), avg_traceroute[src.sprintf("%IP.dst%")], time]
                pre_df_array.append(to_be_added)

    df = pandas.DataFrame(pre_df_array, columns=["ip", "avg traceroute", "ping time"])

    # print into file
    if pre_df_array is not None:
        if return_type=="file":
            filename=f"{savedir}/{name_prefix}_{ip}_{datetime.now().strftime('%Y_%m_%d_%H_%M')}.csv"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            df.to_csv(filename)
        elif return_type=="df":
            return df

def measure_all(ip_file, **kw_args):
    """
    measures ping times for all IPs contained in ip_file
    """
    ips = open(ip_file).readlines()
    for ip in ips:
        measure(ip.replace(" ", "").replace("\n", ""), **kw_args)


