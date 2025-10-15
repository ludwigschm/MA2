### Skript 1 für Kombinationen ###

rm(list = ls())
setwd("C:/Users/ludwi/IONOS HiDrive/users/fincova/studium/Master/Masterarbeit/Experiment")
library(dplyr)

df <- read.csv("Kombinationen.csv", sep = ";", encoding = "UTF-8")
colnames(df)[1] <- "Kategorie"



# Auswählen Anzahl je Kategorie  ------------------------------------------
n <- 16 # Vorgegeben, Anzahl an Trials 
Summen <- aggregate(Wkeit ~ Kategorie, data = df, sum)
Summen$Anzahl <-round(Summen[,2]*n, 0)
sum(Summen$Anzahl)

if (sum(Summen$Anzahl) > n){
  Summen$Anzahl[which.max(Summen$Anzahl)] <- Summen$Anzahl[which.max(Summen$Anzahl)] - 1
} else if (sum(Summen$Anzahl) < n){
  Summen$Anzahl[which.min(Summen$Anzahl)] <- Summen$Anzahl[which.min(Summen$Anzahl)] + 1
} else {
  
}
sum(Summen$Anzahl)


df <- df %>%
  left_join(Summen %>% select(Kategorie, Anzahl), by = "Kategorie")

df$Zeilennummer <- 1:nrow(df)



# 1: Kombinationen ziehen ----------------------------------------------------

cats <- unique(df$Kategorie)
all_trials <- c()   # leerer Vektor zum Sammeln


# Für Spieler 1
set.seed(123)  # für Reproduzierbarkeit
for (kat in cats) {
  # Anzahl Ziehungen für diese Kategorie
  n <- Summen$Anzahl[Summen$Kategorie == kat]
  
  # Ziehen mit Zurücklegen & gewichteter Wahrscheinlichkeit
  trials <- sample(
    df$Zeilennummer[df$Kategorie == kat],
    size = n,
    replace = TRUE,
    prob  = df$Bed..Wkeit[df$Kategorie == kat]
  )
  
  # Ergebnisse an Vektor anhängen
  all_trials <- c(all_trials, trials)
}

all_trials

# Zeilen nun zuordnen
new_df <- df[match(all_trials, df$Zeilennummer), ]


Karten1 <- new_df[, 1:4]
Karten1$Wert <- Karten1$Hand
Karten1$Wert[Karten1$Kategorie == "über"] <- 0 


# Für Spieler 2
all_trials2 <- c()   # leerer Vektor zum Sammeln

set.seed(142)  # für Reproduzierbarkeit
for (kat in cats) {
  # Anzahl Ziehungen für diese Kategorie
  n <- Summen$Anzahl[Summen$Kategorie == kat]
  
  # Ziehen mit Zurücklegen & gewichteter Wahrscheinlichkeit
  trials <- sample(
    df$Zeilennummer[df$Kategorie == kat],
    size = n,
    replace = TRUE,
    prob  = df$Bed..Wkeit[df$Kategorie == kat]
  )
  
  # Ergebnisse an Vektor anhängen
  all_trials2 <- c(all_trials2, trials)
}

all_trials2

# Zeilen nun zuordnen
new_df2 <- df[match(all_trials2, df$Zeilennummer), ]


Karten2 <- new_df2[, 1:4]
Karten2$Wert <- Karten2$Hand
Karten2$Wert[Karten2$Kategorie == "über"] <- 0 

# Namen anpassen
names(Karten1) <- paste0(names(Karten1),"1")
names(Karten2) <- paste0(names(Karten2), "2")






# 1: Karten mischen: ---------------------------------------------------------
x <- 123

repeat {
  set.seed(x) # für Reproduzierbarkeit
  # Zufällige Reihenfolge für Karten2 erzeugen
  Karten2_shuffled <- Karten2[sample(nrow(Karten2)), ]
  
  
  # Zeilenweise zusammenführen
  Paare <- cbind(Karten1, Karten2_shuffled)
  
  # 1, wenn SP1 gewinnt, -1 wenn Sp2 gewinnt
  Paare$Spw <- 0
  for(i in 1:nrow(Paare)){
    if (Paare$Wert1[i] > Paare$Wert2[i]){
      Paare$Spw[i] <- 1
    } else if (Paare$Wert1[i] < Paare$Wert2[i]){
      Paare$Spw[i] <- -1
    } else {
    }
  }
  
  # Spieler 1 & 2 gewinnen gleich oft, 4 mal ist es unentschieden
  if (sum(Paare$Spw) == 0 && sum(Paare$Spw == 0) == 2) {
    break   # Schleife beenden
    } else {
      x <- x + 1   # hochzählen und weiterprobieren
    }
}


sum(Paare$Spw)
x
Paare

set.seed(132)
Paare <- Paare[sample(nrow(Paare)), ]


# 1: Daten als CSV speichern -------------------------------------------------
write.csv(
  Paare,
  file = "Paare1.csv",
  fileEncoding = "UTF-8"
)


# 3: 1 invertieren --------------------------------------------------------
Paare3 <- Paare

set.seed(132)
names(Paare3) <- sub("1$", "X", names(Paare3))  # nur am Ende: 1 → X
names(Paare3) <- sub("2$", "1", names(Paare3))  # nur am Ende: 2 → 1
names(Paare3) <- sub("X$", "2", names(Paare3))
Paare3

write.csv(
  Paare3,
  file = "Paare3.csv",
  fileEncoding = "UTF-8"
)











# 2: Kombinationen ziehen ----------------------------------------------------

cats <- unique(df$Kategorie)
all_trials <- c()   # leerer Vektor zum Sammeln


# Für Spieler 1
set.seed(323)  # für Reproduzierbarkeit
for (kat in cats) {
  # Anzahl Ziehungen für diese Kategorie
  n <- Summen$Anzahl[Summen$Kategorie == kat]
  
  # Ziehen mit Zurücklegen & gewichteter Wahrscheinlichkeit
  trials <- sample(
    df$Zeilennummer[df$Kategorie == kat],
    size = n,
    replace = TRUE,
    prob  = df$Bed..Wkeit[df$Kategorie == kat]
  )
  
  # Ergebnisse an Vektor anhängen
  all_trials <- c(all_trials, trials)
}

all_trials

# Zeilen nun zuordnen
new_df <- df[match(all_trials, df$Zeilennummer), ]


Karten1 <- new_df[, 1:4]
Karten1$Wert <- Karten1$Hand
Karten1$Wert[Karten1$Kategorie == "über"] <- 0 


# Für Spieler 2
all_trials2 <- c()   # leerer Vektor zum Sammeln

set.seed(242)  # für Reproduzierbarkeit
for (kat in cats) {
  # Anzahl Ziehungen für diese Kategorie
  n <- Summen$Anzahl[Summen$Kategorie == kat]
  
  # Ziehen mit Zurücklegen & gewichteter Wahrscheinlichkeit
  trials <- sample(
    df$Zeilennummer[df$Kategorie == kat],
    size = n,
    replace = TRUE,
    prob  = df$Bed..Wkeit[df$Kategorie == kat]
  )
  
  # Ergebnisse an Vektor anhängen
  all_trials2 <- c(all_trials2, trials)
}

all_trials2

# Zeilen nun zuordnen
new_df2 <- df[match(all_trials2, df$Zeilennummer), ]


Karten2 <- new_df2[, 1:4]
Karten2$Wert <- Karten2$Hand
Karten2$Wert[Karten2$Kategorie == "über"] <- 0 

# Namen anpassen
names(Karten1) <- paste0(names(Karten1),"1")
names(Karten2) <- paste0(names(Karten2), "2")






# 2: Karten mischen: ---------------------------------------------------------
x <- 223

repeat {
  set.seed(x) # für Reproduzierbarkeit
  # Zufällige Reihenfolge für Karten2 erzeugen
  Karten2_shuffled <- Karten2[sample(nrow(Karten2)), ]
  
  
  # Zeilenweise zusammenführen
  Paare2 <- cbind(Karten1, Karten2_shuffled)
  
  # 1, wenn SP1 gewinnt, -1 wenn Sp2 gewinnt
  Paare2$Spw <- 0
  for(i in 1:nrow(Paare)){
    if (Paare2$Wert1[i] > Paare2$Wert2[i]){
      Paare2$Spw[i] <- 1
    } else if (Paare2$Wert1[i] < Paare2$Wert2[i]){
      Paare2$Spw[i] <- -1
    } else {
    }
  }
  
  # Spieler 1 & 2 gewinnen gleich oft, 2 mal ist es unentschieden
  if (sum(Paare2$Spw) == 0 && sum(Paare2$Spw == 0) == 2) {
    break   # Schleife beenden
  } else {
    x <- x + 1   # hochzählen und weiterprobieren
  }
}


sum(Paare2$Spw)
x
Paare2

set.seed(132)
Paare2 <- Paare2[sample(nrow(Paare2)), ]

# 2: Daten als CSV speichern -------------------------------------------------
write.csv(
  Paare2,
  file = "Paare2.csv",
  fileEncoding = "UTF-8"
)



# 4: 2 invertieren --------------------------------------------------------
Paare4 <- Paare2
set.seed(132)
names(Paare4) <- sub("1$", "X", names(Paare4))  # nur am Ende: 1 → X
names(Paare4) <- sub("2$", "1", names(Paare4))  # nur am Ende: 2 → 1
names(Paare4) <- sub("X$", "2", names(Paare4))
Paare4

write.csv(
  Paare4,
  file = "Paare4.csv",
  fileEncoding = "UTF-8"
)







# Testtrials --------------------------------------------------------------
df_test <- read.csv("Kombi_test.csv", sep = ";")

write.csv(
  df_test,
  file = "Paaretest.csv",
  fileEncoding = "UTF-8"
)


