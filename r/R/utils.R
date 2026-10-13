#' TBD
#' @keywords internal
clip_prob <- function (x, eps = 1e-6) {

    pmin(pmax(x, eps), 1 - eps)

}

#' TBD
#' @keywords internal
expected_agreement_from_prob <- function (p) {

    mean(p^2 + (1 - p)^2, na.rm = TRUE)

}

#' TBD
#' @keywords internal
safe_glm_fit_predict <- function (train_data, test_data, formula, link_function) {

    fit <- tryCatch (
        stats::glm(
            formula = formula,
            data = train_data,
            family = stats::binomial(link = link_function)
            ),
        error = function (e) NULL
        )

    if (is.null(fit) ) {

        return (
            rep(
                mean(train_data$choice_stim2, na.rm = TRUE),
                nrow(test_data)
                )
            )

    }

    pred <- tryCatch (
        stats::predict(
            fit,
            newdata = test_data,
            type = "response"
            ),
        error = function (e) {
            rep(
                mean(train_data$choice_stim2, na.rm = TRUE),
                nrow(test_data)
                )
        }

        )

    return (clip_prob(as.numeric(pred) ) )

}
