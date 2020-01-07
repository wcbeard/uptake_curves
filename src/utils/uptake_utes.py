def maj_os(dvers):
    vers = str(dvers).split(".")[0]
    try:
        return int(vers)
    except ValueError:
        return
