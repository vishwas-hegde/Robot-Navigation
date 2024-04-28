import task3

# Take average covariance from the 7 covariance matrices

cov1 = task3.estimate_covariances("data/studentdata1.mat")
cov2 = task3.estimate_covariances("data/studentdata2.mat")
cov3 = task3.estimate_covariances("data/studentdata3.mat")
cov4 = task3.estimate_covariances("data/studentdata4.mat")
cov5 = task3.estimate_covariances("data/studentdata5.mat")
cov6 = task3.estimate_covariances("data/studentdata6.mat")
cov7 = task3.estimate_covariances("data/studentdata7.mat")

cov = (cov1 + cov2 + cov3 + cov4 + cov5 + cov6 + cov7) / 7

print(cov)