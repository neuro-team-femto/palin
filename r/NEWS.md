# palin 0.0.7

* Allowing the user to change the upper bound on internal noise during estimation (see `?sdt_fitting()`).
* The `sdt_fitting()` function can now use a multinomial likelihood function/loss (instead of the MSE loss).
* The `sdt_fitting()` function now returns profile confidence interval around bias and internal noise estimates.

# palin 0.0.6

* Fixing the GAM smoothing in `sdt_fitting()` and now plotting the fitted surface.
* Fixing an error in `sdt_data()` (stim2 should be chosen when signed evidence is positive).
* Now allowing to return signed evidence on every trial.

# palin 0.0.5

* Fixing a few errors in `computing_kernel()` and `response_consistency()`.
* Improving the `template_distance` method in `response_consistency()` (now considering the ratio of distances ratio).

# palin 0.0.4

* Improving the SDT fit methods (more efficient `DEoptim` algorithm, smoothing the error surface).
* Improving the GLMM method for computing the kernel (now using a Bayesian GLMM).
* Including a semi-analytical solution for computing prop_agree and prop_first in `sdt_data()`.
* Implementing the `intercept` method (using a binomial GLM) in `response_consistency()`.

# palin 0.0.3

* Increasing the flexibility of input data format by allowing the user to specify the relevant columns.
* New methods for computing response consistency (distance to templates or similarity to kernel).
* Improving the SDT fit methods.
* Improving the GLM method for computing the kernel (now returning the CI).

# palin 0.0.2

* Including a `NEWS.md` file to track changes.
* Including the (more efficient) `DEoptim` method for fitting the DDM and SDT model.
* Including the `response_consistency()` function to compute response consistency.
* Reworking the GLMM method for estimating the kernel (and internal noise), now based on `mgcv::bam()`.
* Reworking the `README.md` file with a full comparison of estimates from the DDM and SDT model.

# palin 0.0.1

* Pushing the first `R` version of `palin`.
