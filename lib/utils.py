def get_eta(start, end, extra, num_left):
    exe_s = end - start
    eta_s = (exe_s + extra) * num_left
    eta = {'h': 0, 'm': 0, 's': 0}
    if eta_s < 60:
        eta['s'] = int(eta_s)
    elif eta_s >= 60 and eta_s < 3600:
        eta['m'] = int(eta_s / 60)
        eta['s'] = int(eta_s % 60)
    else:
        eta['h'] = int(eta_s / (60 * 60))
        eta['m'] = int(eta_s % (60 * 60) / 60)
        eta['s'] = int(eta_s % (60 * 60) % 60)

    return eta