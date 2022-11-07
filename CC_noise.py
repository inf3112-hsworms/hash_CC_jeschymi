import hashlib
from collections import Counter
from datetime import datetime
import random


def add_cc_noise(data, hashfunc=hashlib.sha256, file="alice"):
    """
    This function simulates the sending of a message with text that follows
    the same character probability distribution as the text contained in file.
    The message is sent over a covert channel as described by Schmidbauer and Wendzel.
    This function returns the modified data as well as the additional channel times alone.
    """
    result = []
    alice = open(file).read()

    most_common = [a[0] for a in Counter(alice).most_common()]

    channel = []

    for r in data:
        char = random.choice(alice)
        time_ = datetime.now()
        for m in most_common:
            hashfunc(bytes(alice + char + str(time_), "UTF-8"))
            if m == char:
                break
        time = (datetime.now() - time_).total_seconds()
        result.append(r + time)
        channel.append(time)

    return result, channel
