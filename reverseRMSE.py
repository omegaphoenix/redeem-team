# Inputs
fakeRMSE = 0.90591
multiplier = .96479
cinematch = .9514

RMSE = fakeRMSE * multiplier
print RMSE
print (cinematch - RMSE) / cinematch
