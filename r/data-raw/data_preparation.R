# code to prepare the self-produced speech data
self_produced_speech <- read.csv(file = here::here("./data-raw/self_produced_speech.csv") )
usethis::use_data(self_produced_speech, overwrite = TRUE)

# code to prepare the (full version of the) self-produced speech data
self_produced_speech_full <- read.csv(file = here::here("./data-raw/self_produced_speech_full.csv") )
usethis::use_data(self_produced_speech_full, overwrite = TRUE)
