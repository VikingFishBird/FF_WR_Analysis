import datetime

print(datetime.datetime.now())

#import pandas as pd
import numpy as np
#import matplotlib as mplt
#import matplotlib.pyplot as plt
from sportsreference.nfl.teams import Teams
from sportsreference.nfl.roster import Roster
from sportsreference.nfl.player import AbstractPlayer

#p = AbstractPlayer()
#p.times_pass_target


pass_catchers = []

print(datetime.datetime.now())


teams = Teams(2019)
for team in teams:
    roster = team.roster

    for player in roster.players:
        try:
            if player is not None and player.times_pass_target is not None and int(player.times_pass_target) > 10:
                pass_catchers.append(player.player_id)
        except:  # Errors with offensive lineman
            continue
    break

print(datetime.datetime.now())

print(pass_catchers)
