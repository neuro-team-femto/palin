#' Computing the kernel from 2AFC data
#'
#' Computing the kernel from 2AFC data. It should be noted that when using the
#' "glm" method, the group-level (average) kernel is simply averaged from the
#' individual-level kernels. To obtain **shrunk** (i.e., adjusted) group-level
#' kernels, one needs to specify the "glmm" method.
#'
#' @param data Dataframe, with reverse correlation data. Should contain the
#' following columns: participant, block, trial, resp, f, eq.
#' @param method Character, the kernel computation method, either "difference", "glm", or "glmm".
#' @param double_pass Logical, indicating whether the last block was repeated.
#'
#' @return The original data augmented with the kernel.
#'
#' @importFrom rlang .data
#' @importFrom stats coef
#'
#' @examples
#' \dontrun{
#' # importing the self-voice data
#' data(self_voice)
#' head(self_voice)
#'
#' # computing the kernel per participant
#' computing_kernel(data = self_voice, method = "difference")
#'
#' # plotting it
#' computing_kernel(data = self_voice, method = "difference") |> plot()
#' computing_kernel(data = self_voice, method = "glm") |> plot()
#' }
#'
#' @author Ladislas Nalborczyk \email{ladislas.nalborczyk@@gmail.com}.
#'
#' @export

computing_kernel <- function (
        data, method = c("difference", "glm", "glmm"),
        double_pass = TRUE
        ) {

    # ensuring that the method is one of the above
    method <- match.arg(method)

    # some tests for variable types
    stopifnot("data must be a dataframe..." = is.data.frame(data) )
    stopifnot("method must be a character..." = is.character(method) )
    stopifnot("double_pass must be a logical..." = is.logical(double_pass) )

    # if double-pass was used, remove the last (repeated) block
    if (double_pass) data <- data |> dplyr::filter(.data$block != max(.data$block) )

    if (method == "difference") {

        # first-order kernel computed for each subject as
        # mean(filter gains of voices classified as mine) - mean(filter gains of voices classified as not-mine)
        kernel <- data |>
            # grouping by participant, frequency band, and response
            dplyr::group_by(.data$participant, .data$f, .data$resp) |>
            # computing the average gain
            dplyr::summarise(filter_mean_gain = mean(.data$eq) ) |>
            dplyr::ungroup() |>
            tidyr::pivot_wider(names_from = .data$resp, values_from = .data$filter_mean_gain) |>
            # renaming the columns
            dplyr::rename(negative = .data$`0`, positive = .data$`1`) |>
            # computing the kernel
            # dplyr::filter(.data$f != 0) |>
            dplyr::mutate(kernel_gain = .data$positive - .data$negative) |>
            # kernels are then normalized for each participant by dividing them
            # by the square root of the sum of their squared values
            dplyr::group_by(.data$participant) |>
            # as in the code of Ponsot et al. smile paper
            dplyr::mutate(norm_gain1 = .data$kernel_gain / sqrt(mean(.data$kernel_gain^2) ) ) |>
            # as in the Ponsot et al. PNAS paper and Goupil et al. NatureCom paper
            dplyr::mutate(norm_gain2 = .data$kernel_gain / sum(abs(.data$kernel_gain) ) ) |>
            dplyr::ungroup()

        # setting the class of the resulting object
        class(kernel) <- c("kernel", "data.frame")

    } else if (method == "glm") {

        # fitting a generalised (binomial) linear model
        model <- stats::glm(
            formula = resp ~ 1 + eq * f,
            family = stats::binomial(),
            data = data |> dplyr::mutate(f = as.factor(.data$f) )
            )

        # extracting and reshaping the coefficients
        coefficients <- data.frame(stats::coef(model) ) |>
            dplyr::mutate(rowname = rownames(data.frame(stats::coef(model) ) ) ) |>
            dplyr::filter(stringr::str_detect(string = .data$rowname, pattern = ":") ) |>
            dplyr::pull(.data$stats..coef.model.)

        # taking into account the average affect of eq
        coefficients <- coefficients + stats::coef(model)[["eq"]]

        kernel <- data |>
            # grouping by participant, frequency band, and response
            dplyr::group_by(.data$participant, .data$f, .data$resp) |>
            # computing the average gain
            dplyr::summarise(filter_mean_gain = mean(.data$eq) ) |>
            dplyr::ungroup() |>
            tidyr::pivot_wider(names_from = .data$resp, values_from = .data$filter_mean_gain) |>
            # computing the kernel
            dplyr::filter(.data$f != 0) |>
            dplyr::mutate(kernel_gain = rep(coefficients, dplyr::n_distinct(data$participant) ) ) |>
            # kernels are then normalized for each participant by dividing them
            # by the square root of the sum of their squared values
            dplyr::group_by(.data$participant) |>
            # as in the code of Ponsot et al. smile paper
            dplyr::mutate(norm_gain1 = .data$kernel_gain / sqrt(mean(.data$kernel_gain^2) ) ) |>
            # as in the Ponsot et al. PNAS paper and Goupil et al. NatureCom paper
            dplyr::mutate(norm_gain2 = .data$kernel_gain / sum(abs(.data$kernel_gain) ) ) |>
            dplyr::ungroup()

        # setting the class of the resulting object
        class(kernel) <- c("kernel", "data.frame")

    } else if (method == "glmm") {

        # reshaping the data
        data2 <- data |>
            # focusing on speech-relevant frequency bands
            dplyr::filter(.data$f > 100) |>
            dplyr::filter(.data$f < 10000) |>
            # removing the double-pass block (if needed)
            # {if (double_pass) dplyr::filter(., .data$block < max(.data$block) ) else .} |>
            dplyr::filter(if (double_pass) .data$block < max(.data$block) else TRUE) |>
            dplyr::mutate(f = as.factor(.data$f) ) |>
            dplyr::mutate(participant = as.factor(.data$participant) ) |>
            # converting to data table
            data.table::as.data.table()

        # fitting a multilevel generalised (binomial) linear model per participant
        # this does not work for the moment...
        # model <- lme4::glmer(
        #     formula = resp ~ 1 + eq * f + (1 + eq | participant) + (1 + eq | participant:f),
        #     family = stats::binomial(),
        #     data = data2
        #     )
        # model2 <- lme4::glmer(
        #     formula = resp ~ 1 + eq * f + (1 + eq * f | participant),
        #     family = stats::binomial(),
        #     data = data |> dplyr::mutate(f = as.factor(.data$f) )
        #     )

        # using a GAM instead
        # see https://m-clark.github.io/posts/2019-10-20-big-mixed-models/
        gamm <- mgcv::bam(
            formula = resp ~ 1 + eq * f +
                s(participant, bs = "re") +
                s(eq, f, participant, bs = "re"),
                # s(participant, by = eq, bs = "re"),
                # s(participant, by = f, bs = "re") +
                # s(participant, by = interaction(eq, f), bs = "re"),
            family = stats::binomial(link = "logit"),
            data = data2,
            discrete = TRUE,
            # https://stackoverflow.com/questions/78590558/how-can-i-enable-openmp-in-mac-os-for-mgcv-r-package
            nthreads = parallel::detectCores() - 1,
            method = "fREML"
            )

        ############################
        # retrieving fixed effects #
        ############################

        # retrieving model coefficients
        coefficients <- stats::coef(gamm)

        # main effect of eq
        eq_coef <- coefficients["eq"]

        # interaction terms
        eq_f_interaction <- coefficients[grepl("^eq:f", names(coefficients) )]

        # creating a table of slopes for eq at each f
        group_kernel <- data.frame(
            f = sub("eq:f", "", names(eq_f_interaction) ),
            kernel_gain = eq_coef + eq_f_interaction
            ) |>
            dplyr::mutate(participant = "group") |>
            dplyr::relocate(.data$participant, 1)

        # removing rownames
        rownames(group_kernel) <- NULL

        # generating a grid of possible values for predictions
        # new_data <- expand.grid(
        #     eq = unique(data2$eq),
        #     f = unique(data2$f),
        #     participant = NA
        #     )

        # retrieving predictions with SE
        # new_data$predicted <- predict(
        #     gamm, newdata = new_data, type = "link",
        #     exclude = "s(participant)",
        #     se.fit = TRUE
        #     )

        #############################
        # retrieving random effects #
        #############################

        # creating dataset with small variations in eq to estimate slopes
        new_data <- tidyr::crossing(
            participant = unique(data2$participant),
            f = unique(data2$f)
            ) |>
            dplyr::mutate(
                # small perturbation below mean eq
                eq_low = mean(data2$eq) - 0.01,
                # small perturbation above mean eq
                eq_high = mean(data2$eq) + 0.01
                )

        # predicting values at slightly different eq levels
        new_data$pred_low <- stats::predict(
            gamm, newdata = dplyr::mutate(new_data, eq = .data$eq_low),
            type = "link"
            )
        new_data$pred_high <- stats::predict(
            gamm, newdata = dplyr::mutate(new_data, eq = .data$eq_high),
            type = "link"
            )

        # computing approximate slope (kernel) as finite difference
        kernel <- new_data |>
            dplyr::mutate(
                kernel_gain = (.data$pred_high - .data$pred_low) /
                    (.data$eq_high - .data$eq_low)
                ) |>
            dplyr::select(.data$participant, .data$f, .data$kernel_gain)

        # appending fixed and random effects
        kernel <- dplyr::bind_rows(kernel, group_kernel)

        # setting the class of the resulting object
        class(kernel) <- c("multilevel_kernel", "data.frame")

    # } else if (method == "bglmm") {
    #
    #     # reshaping the data
    #     data2 <- data |>
    #         # focusing on speech-relevant frequency bands
    #         dplyr::filter(.data$f > 100) |>
    #         dplyr::filter(.data$f < 10000) |>
    #         # removing the double-pass block (if needed)
    #         # {if (double_pass) dplyr::filter(., .data$block < max(.data$block) ) else .} |>
    #         dplyr::filter(if (double_pass) .data$block < max(.data$block) else TRUE) |>
    #         dplyr::mutate(f = as.factor(.data$f) ) |>
    #         dplyr::mutate(participant = as.factor(.data$participant) ) |>
    #         # converting to data table
    #         data.table::as.data.table()
    #
    #     # Bayesian GLMM approach with participant as varying effect
    #     bglmm <- brms::brm(
    #         formula = resp ~ 1 + eq * f + (1 | participant),
    #         family = brms::bernoulli(),
    #         data = data2,
    #         chains = 4,
    #         cores = 4
    #         )

    }

    # returning it
    return (kernel)

}

#' @export

plot.kernel <- function (
        x, normalisation_method = c("kernel_gain", "norm_gain1", "norm_gain2"),
        ...
        ) {

    # ensuring that the normalisation method is one of the above
    normalisation_method <- match.arg(normalisation_method)

    # plotting the filters
    x |>
        # keeping only the relevant frequency bands for human speech
        # dplyr::filter(.data$f > 0 & .data$f < 10000) |>
        # dplyr::filter(.data$f > 0) |>
        # plotting it
        ggplot2::ggplot(
            ggplot2::aes(
                x = .data$f,
                y = .data$kernel_gain,
                colour = .data$participant,
                fill = .data$participant
                )
            ) +
        ggplot2::geom_hline(yintercept = 0, linetype = 2) +
        # plotting each participant's filter
        ggplot2::geom_line(size = 0.5, alpha = 0.3, show.legend = FALSE) +
        # plotting the average filter
        ggplot2::stat_summary(
            ggplot2::aes(x = .data$f, y = .data[[normalisation_method]]),
            geom = "line",
            fun = mean,
            size = 1,
            inherit.aes = FALSE
            ) +
        # aesthetics
        ggplot2::theme_bw(base_family = "Open Sans", base_size = 12) +
        ggplot2::scale_colour_manual(
            values = MetBrewer::met.brewer(name = "Johnson", n = dplyr::n_distinct(x$participant) )
            ) +
        ggplot2::scale_fill_manual(
            values = MetBrewer::met.brewer(name = "Johnson", n = dplyr::n_distinct(x$participant) )
            ) +
        {if (normalisation_method == "kernel_gain") ggplot2::labs(
            x = "Frequency (Hz)",
            y = "Filter amplitude (a.u.)"
            )} +
        {if (normalisation_method != "kernel_gain") ggplot2::labs(
            x = "Frequency (Hz)",
            y = "Normalised filter amplitude (a.u.)"
            )} +
        ggplot2::scale_x_log10(
            breaks = scales::breaks_log(n = 5, base = 10),
            labels = scales::label_log(base = 10)
            ) +
        ggplot2::annotation_logticks(sides = "b")

}

#' @export

plot.multilevel_kernel <- function (x, ...) {

    x |>
        ggplot2::ggplot(
            ggplot2::aes(
                x = as.numeric(.data$f),
                y = .data$kernel_gain,
                group = .data$participant,
                color = .data$participant
                )
            ) +
        ggplot2::geom_hline(yintercept = 0, linetype = 2) +
        # plotting individual kernels
        ggplot2::geom_line(
            data = x |> dplyr::filter(.data$participant != "group"),
            linewidth = 0.5, alpha = 0.5
            ) +
        # plotting the group-level average kernel
        # plotting individual kernels
        ggplot2::geom_line(
            data = x |> dplyr::filter(.data$participant == "group"),
            linewidth = 1
            ) +
        ggplot2::labs(
            x = "Frequency (Hz)",
            y = "Filter amplitude (a.u.)"
            ) +
        # aesthetics
        ggplot2::theme_bw(base_family = "Open Sans", base_size = 12) +
        ggplot2::scale_colour_manual(
            values = MetBrewer::met.brewer(
                name = "Johnson",
                n = dplyr::n_distinct(x$participant)
                )
            ) +
        ggplot2::theme(legend.position = "none") +
        ggplot2::scale_x_log10(
            breaks = scales::breaks_log(n = 5, base = 10),
            labels = scales::label_log(base = 10)
            ) +
        ggplot2::annotation_logticks(sides = "b")

}
