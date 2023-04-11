#' Computing the kernel from 2AFC data
#'
#' Computing the kernel from 2AFC data. It should be noted that when using the
#' "glm" method, the group-level (average) kernel is simply averaged from the
#' individual-level kernels. To obtain **shrinked** (i.e., adjusted) group-level
#' kernels, one needs to specify the "glmm" method (work in progress).
#'
#' @param data Dataframe, containing the data...
#' @param method Character, the computation method, either "difference", "glm", or "glmm".
#' @param double_pass Logical, indicating whether the last block was repeated.
#'
#' @return The original data augmented with the kernel.
#'
#' @importFrom rlang .data
#'
#' @examples
#' \dontrun{
#' # importing the data
#' data(self_produced_speech_full)
#' head(self_produced_speech_full)
#'
#' # computing the kernel
#' computing_kernel(data = self_produced_speech_full, method = "difference")
#'
#' # plotting it
#' computing_kernel(data = self_produced_speech_full, method = "difference") |> plot()
#' computing_kernel(data = self_produced_speech_full, method = "glm") |> plot()
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
    if (double_pass) data <-  data |> dplyr::filter(.data$block != max(.data$block) )

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
            # computing the kernel
            dplyr::mutate(kernel_gain = .data$`1` - .data$`0`) |>
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

        # extracting the coefficients
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

        # fitting a multilevel generalised (binomial) linear model
        model <- lme4::glmer(
            formula = resp ~ 1 + eq + (1 + eq | f),
            family = stats::binomial(),
            data = data |> dplyr::mutate(f = as.factor(.data$f) )
            )

        # the code below does not work for the moment...
        # model <- lme4::glmer(
        #     formula = resp ~ 1 + eq * f + (1 | participant),
        #     family = stats::binomial(),
        #     data = data |> dplyr::mutate(f = as.factor(.data$f) )
        #     )

        # extracting the coefficients
        coefficients <- lme4::ranef(model)$f$eq

        kernel <- data |>
            # grouping by participant, frequency band, and response
            dplyr::group_by(.data$participant, .data$f, .data$resp) |>
            # computing the average gain
            dplyr::summarise(filter_mean_gain = mean(.data$eq) ) |>
            dplyr::ungroup() |>
            tidyr::pivot_wider(names_from = .data$resp, values_from = .data$filter_mean_gain) |>
            # computing the kernel
            dplyr::mutate(kernel_gain = rep(coefficients, 2) ) |>
            # kernels are then normalized for each participant by dividing them
            # by the square root of the sum of their squared values
            dplyr::group_by(.data$participant) |>
            # as in the code of Ponsot et al. smile paper
            dplyr::mutate(norm_gain1 = .data$kernel_gain / sqrt(mean(.data$kernel_gain^2) ) ) |>
            # as in the Ponsot et al. PNAS paper and Goupil et al. NatureCom paper
            dplyr::mutate(norm_gain2 = .data$kernel_gain / sum(abs(.data$kernel_gain) ) ) |>
            dplyr::ungroup()

        # setting the class of the resulting object
        class(kernel) <- c("multilevel_kernel", "data.frame")

    }

    # returning it
    return (kernel)

}

#' @export

plot.kernel <- function (
        x, normalisation_method = c("kernel_gain", "norm_gain1", "norm_gain2"), ...
        ) {

    # ensuring that the normalisation method is one of the above
    normalisation_method <- match.arg(normalisation_method)

    x |>
        # keeping only the relevant frequency bands for human speech
        dplyr::filter(.data$f > 0 & .data$f < 10000) |>
        # plotting it
        ggplot2::ggplot(
            ggplot2::aes(
                x = .data$f, y = .data$kernel_gain,
                colour = .data$participant, fill = .data$participant
                )
            ) +
        ggplot2::geom_hline(yintercept = 0, linetype = 3) +
        # plotting each participant's filter
        ggplot2::geom_line(size = 0.75, alpha = 0.5) +
        # plotting the average filter
        ggplot2::stat_summary(
            ggplot2::aes(x = .data$f, y = .data[[normalisation_method]]),
            geom = "line", fun = mean,
            size = 1,
            inherit.aes = FALSE
            ) +
        # aesthetics
        ggplot2::theme_bw(base_family = "Open Sans", base_size = 12) +
        ggplot2::scale_color_brewer(palette = "Dark2") +
        ggplot2::scale_fill_brewer(palette = "Dark2") +
        {if (normalisation_method == "kernel_gain") ggplot2::labs(
            x = "Frequency (Hz)",
            y = "Filter amplitude (units of external noise SD)"
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
