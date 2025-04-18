SK_COSTS = [11589810794.81081, 4249478109.0, 313933100.0, 5674976956.0, 553318340.0, 306945894873.0, 10811215372.0]
SK_MODEL_COSTS = [7712716663.418919, 2877093309.0, 313152940.0, 5239383524.0, 417914300.0, 306078443097.0, 1302966448.0]


for i in range(len(SK_COSTS)):
    difference = SK_COSTS[i] - SK_MODEL_COSTS[i]
    print(f"Difference for index {i}: {difference}")
    percentage_difference = (difference / SK_COSTS[i]) * 100 if SK_COSTS[i] != 0 else 0
    print(f"Percentage difference for index SK {i}: {percentage_difference:.2f}%")


SAMSUNG_COSTS = [14897199596.0, 4395009009.0, 1934195500, 31382422620, 2278080036, 1771596908343.0, 54958952372.0]
SAMSUNG_MODEL_COSTS = [12992555628.0, 2957418129.0, 328859220.0, 5358023124.0, 720435844.0, 315465573135.0, 1418333628.0]

for i in range(len(SAMSUNG_COSTS)):
    difference = SAMSUNG_COSTS[i] - SAMSUNG_MODEL_COSTS[i]
    print(f"Difference for index {i}: {difference}")
    percentage_difference = (difference / SAMSUNG_COSTS[i]) * 100 if SAMSUNG_COSTS[i] != 0 else 0
    print(f"Percentage difference for index Samsung{i}: {percentage_difference:.2f}%")