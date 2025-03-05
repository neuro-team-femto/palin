# code to prepare the self-produced speech data
# self_produced_speech <- read.csv(file = here::here("./data-raw/self_produced_speech.csv") )
# usethis::use_data(self_produced_speech, overwrite = TRUE)

# code to prepare (full version of the) self-produced speech data
# self_produced_speech_full <- read.csv(file = here::here("./data-raw/self_produced_speech_full.csv") )
# usethis::use_data(self_produced_speech_full, overwrite = TRUE)

# code to prepare the self-voice data
self_voice <- read.csv(file = here::here("./data-raw/self_voice.csv") ) |>
    # removing a participant who missed a block
    dplyr::filter(participant != unique(participant)[16]) |>
    # keeping only the first 5 participants to reduce the size of the data
    dplyr::filter(participant %in% unique(participant)[1:5])

# exporting to .rda
usethis::use_data(self_voice, overwrite = TRUE, compress = "xz")
