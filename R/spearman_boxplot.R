if (!requireNamespace("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager")
}

# List of required packages
packages <- c(
  'tidyverse', 'readr', 'stringr', 'scales',
  'dplyr', 'tidyr', 'ggplot2', 'patchwork', 'ggpubr'
  )

# Install all missing packages
install_missing <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    tryCatch(
      install.packages(pkg, dependencies = TRUE),
      error = function(e) {
        BiocManager::install(pkg, ask = FALSE, update = FALSE)
      }
    )
  }
}
invisible(sapply(packages, install_missing))

# Load all packages
invisible(lapply(packages, library, character.only = TRUE))

set.seed(100)

path_meta   <- "datasets/df.ngs.meta.csv"
path_pcr   <- "datasets/df.pcr_long.csv"
path_rdp   <- "datasets/df.rdp.genus.csv"
path_silva <- "datasets/df.silva.genus.csv"
path_rdp_f   <- "datasets/df.rdp.csv"
path_silva_f <- "datasets/df.silva.csv"

meta <- read.csv(path_meta, sep = ";", check.names = FALSE)
df_pcr   <- read.csv(path_pcr,   sep = ";", check.names = FALSE)
df_rdp   <- read.csv(path_rdp,   sep = ";", check.names = FALSE)
df_silva <- read.csv(path_silva, sep = ";", check.names = FALSE)
df_rdp_f   <- read.csv(path_rdp_f, sep = ";", check.names = FALSE)
df_silva_f <- read.csv(path_silva_f, sep = ";", check.names = FALSE)

rownames(df_rdp) <-  paste0(meta$Name, '-', meta$Method)
rownames(df_rdp_f) <-  paste0(meta$Name, '-', meta$Method)
rownames(df_silva) <-  paste0(meta$Name, '-', meta$Method)
rownames(df_silva_f) <-  paste0(meta$Name, '-', meta$Method)
df_rdp <- cbind(meta$Name, df_rdp)
df_rdp_f <- cbind(meta$Name, df_rdp_f)
df_silva <- cbind(meta$Name, df_silva)
df_silva_f <- cbind(meta$Name, df_silva_f)
df_rdp_rel <- cbind(df_rdp[,1], df_rdp[,-1]/rowSums(df_rdp[,-1]))
df_rdp_rel_f <- cbind(df_rdp_f[,1], df_rdp_f[,-1]/rowSums(df_rdp_f[,-1]))
df_silva_rel <- cbind(df_silva[,1], df_silva[,-1]/rowSums(df_silva[,-1]))
df_silva_rel_f <- cbind(df_silva_f[,1], df_silva_f[,-1]/rowSums(df_silva_f[,-1]))
df_pcr_rel <- df_pcr
df_pcr_rel[ , 5:ncol(df_pcr)] <- df_pcr_rel[ , 5:ncol(df_pcr)] / df_pcr_rel$PCR_Total
df_silva_rel[,1] <- rownames(df_silva_rel)
df_rdp_rel_f[,1] <- rownames(df_rdp_rel_f)
df_rdp_rel[,1] <- rownames(df_rdp_rel)
df_silva_rel_f[,1] <- rownames(df_silva_rel_f)
#cbind(df_rdp_rel, 'Sample')

## ===== Список таксонов для графика =====
targets_display <- c(
  "Bacteroides sp.","Bifidobacterium sp.","Christensenellaceae","Enterobacteriaceae",
  "Oscillibacter sp.","Faecalibacterium prausnitzii","Lactobacillaceae",
  "Odoribacter sp.","Ruminococcus sp.","Enterococcus faecalis","Subdoligranulum sp."
)

#СПИРМАН ГЕНУС

min_n_pairs <- 2  # минимум пар на род для расчёта Спирмана

# ---------- утилиты ----------
add_sample_col <- function(df) {
  # если первая колонка без имени, считаем, что это SampleID
    names(df)[1] <- "SampleID"
    return(df)
  
  id_candidates <- intersect(names(df), c("SampleID","Name","Sample","Run","ID"))
  if (length(id_candidates) == 0) {
    df <- tibble::rownames_to_column(df, var = "SampleID")
  } else {
    names(df)[names(df) == id_candidates[1]] <- "SampleID"
  }
  df
}

# Нормализация строковых значений чисел (',' → '.')
to_num <- function(x) suppressWarnings(as.numeric(gsub(",", ".", x)))

# Унификация имени рода (для join и агрегации)
normalize_genus_name <- function(x) {
  x <- as.character(x)
  x <- gsub("^g__?", "", x, ignore.case = TRUE)          # g__Bacteroides → Bacteroides
  x <- gsub("^g[_.: ]+", "", x, ignore.case = TRUE)
  x <- gsub("^Genus[_.: ]+", "", x, ignore.case = TRUE)
  x <- gsub("\\[|\\]", "", x)                            # [Eubacterium] → Eubacterium
  x <- gsub("(?i)\\b(group|complex|cluster)\\b", "", x)  # убрать служебные слова
  x <- gsub("(?i)\\b(sp\\.?|spp\\.?)\\b", "", x)         # убрать sp., spp.
  x <- gsub("[^A-Za-z0-9]+", "_", x)                     # всё прочее → _
  x <- gsub("^_+|_+$", "", x)                            # убрать крайние _
  toupper(x)                                             # для надёжного совпадения
}

# PCR: берём колонки PCR_* и трактуем их как роды
to_long_pcr_genus <- function(df) {
  df <- add_sample_col(df)
  taxa_cols <- grep("^PCR_", names(df), value = TRUE)
  taxa_cols <- taxa_cols[!grepl("_?Total$", taxa_cols, ignore.case = TRUE)]
  stopifnot(length(taxa_cols) > 0)
  
  tmp <- df %>% select(SampleID, all_of(taxa_cols))
  clean_names <- gsub("^PCR_", "", names(tmp)[-1])
  key_genus   <- normalize_genus_name(clean_names)
  
  names(tmp) <- c("SampleID", key_genus)
  
  long <- tmp %>%
    pivot_longer(-SampleID, names_to = "Genus_key", values_to = "abundance") %>%
    mutate(abundance = to_num(abundance)) %>%
    filter(is.finite(abundance)) %>%
    group_by(SampleID, Genus_key) %>%
    summarize(abundance = sum(abundance, na.rm = TRUE), .groups = "drop")
  
  # Сделаем красивую подпись для оси (Genus_label) — пробелы вместо _
  long %>%
    mutate(Genus = gsub("_", " ", Genus_key))
}

# RDP/SILVA: берём генусные колонки
to_long_ngs_genus <- function(df) {
  df <- add_sample_col(df)
  # сначала явные генусные префиксы
  genus_cols <- grep("^(g__|g[_.: ]|Genus[_.: ])", names(df), value = TRUE, ignore.case = TRUE)
  
  # если не нашли — возьмём колонки, которые НЕ начинаются с f_/o_/c_/p_/k_
  if (length(genus_cols) == 0) {
    genus_cols <- names(df)[
      !names(df) %in% "SampleID" &
        !grepl("^[focpk]_", names(df), ignore.case = TRUE)
    ]
  }
  stopifnot(length(genus_cols) > 0)
  
  tmp <- df %>% select(SampleID, all_of(genus_cols))
  key_genus <- normalize_genus_name(names(tmp)[-1])
  names(tmp) <- c("SampleID", key_genus)
  
  long <- tmp %>%
    pivot_longer(-SampleID, names_to = "Genus_key", values_to = "abundance") %>%
    mutate(abundance = to_num(abundance)) %>%
    filter(is.finite(abundance)) %>%
    group_by(SampleID, Genus_key) %>%
    summarize(abundance = sum(abundance, na.rm = TRUE), .groups = "drop")
  
  long %>% mutate(Genus = gsub("_", " ", Genus_key))
}

# Спирман по родам
spearman_by_genus <- function(long_a, long_b, label) {
  out <- inner_join(long_a, long_b,
                    by = c("SampleID","Genus_key"),
                    suffix = c("_PCR","_NGS")) %>%
    group_by(Genus_key) %>%
    summarize(
      Genus = dplyr::first(Genus_PCR %||% Genus_NGS),
      n     = n(),
      rho   = if (n() >= min_n_pairs &&
                  sd(abundance_PCR) > 0 && sd(abundance_NGS) > 0)
        suppressWarnings(cor(abundance_PCR, abundance_NGS, method = "spearman"))
      else NA_real_,
      pval  = if (n() >= min_n_pairs &&
                  sd(abundance_PCR) > 0 && sd(abundance_NGS) > 0)
        tryCatch(suppressWarnings(
          cor.test(abundance_PCR, abundance_NGS, method = "spearman")$p.value),
          error = function(e) NA_real_)
      else NA_real_,
      .groups = "drop"
    ) %>%
    ungroup()
  
  # FDR (BH) по всем генусам в рамках этого сравнения
  out <- out %>%
    mutate(
      qval = p.adjust(pval, method = "BH"),
      comparison = label
    )
  
  out
}


`%||%` <- function(a, b) if (!is.null(a) && !all(is.na(a))) a else b
pcr_long   <- to_long_pcr_genus(df_pcr_rel)
rdp_long   <- to_long_ngs_genus(df_rdp_rel)
silva_long <- to_long_ngs_genus(df_silva_rel)

collapse_to_genus <- function(df){
  df %>%
    mutate(
      Genus_key = ifelse(
        grepl("ACEAE$", Genus_key, ignore.case = TRUE),
        toupper(Genus_key),
        toupper(sub("_.*$", "", Genus_key))
      ),
      # подпись для графиков/таблиц: пробелы вместо "_"
      Genus = gsub("_", " ", Genus_key)
    )
}

pcr_long   <- collapse_to_genus(pcr_long)
rdp_long   <- collapse_to_genus(rdp_long)
silva_long <- collapse_to_genus(silva_long)


# --- нормализуем ID для пересечений ---
normalize_id <- function(x) trimws(as.character(x))
pcr_long$SampleID   <- normalize_id(pcr_long$SampleID)
rdp_long$SampleID   <- normalize_id(rdp_long$SampleID)
silva_long$SampleID <- normalize_id(silva_long$SampleID)

# --- пересечения родов ---
gen_pcr_rdp   <- intersect(unique(pcr_long$Genus_key),   unique(rdp_long$Genus_key))
gen_pcr_silva <- intersect(unique(pcr_long$Genus_key), unique(silva_long$Genus_key))
cat("Genera: PCR–RDP =", length(gen_pcr_rdp),
    " PCR–SILVA =", length(gen_pcr_silva), "\n")

# --- Спирман по родам ---
res_rdp <- spearman_by_genus(
  filter(pcr_long, Genus_key %in% gen_pcr_rdp),
  filter(rdp_long, Genus_key %in% gen_pcr_rdp),
  "PCR vs RDP"
)

res_silva <- spearman_by_genus(
  filter(pcr_long,   Genus_key %in% gen_pcr_silva),
  filter(silva_long, Genus_key %in% gen_pcr_silva),
  "PCR vs SILVA"
)

res_silva_rdp <- spearman_by_genus(
  filter(rdp_long,   Genus_key %in% gen_pcr_silva),
  filter(silva_long, Genus_key %in% gen_pcr_silva),
  "RDP vs SILVA"
)

res_all <- bind_rows(res_rdp, res_silva)


family_targets <- c("Enterobacteriaceae","Lactobacillaceae","Christensenellaceae")

to_num <- function(x) suppressWarnings(as.numeric(gsub(",", ".", x)))

# --- PCR: берём точные семейства из PCR_* (без агрегаций) ---
pcr_fam_long <- add_sample_col(df_pcr_rel) %>%
  select(any_of(c("SampleID", paste0("PCR_", family_targets)))) %>%
  pivot_longer(-SampleID, names_to = "PCR_Family", values_to = "abundance") %>%
  mutate(
    Family    = sub("^PCR_", "", PCR_Family),
    abundance = to_num(abundance)
  ) %>%
  filter(Family %in% family_targets, is.finite(abundance)) %>%
  transmute(SampleID, Family, abundance)

# --- RDP/SILVA: берём точные f_* семейства (если в датасете префикс f__, f:, f. — тоже снимется) ---
rdp_fam_long <- add_sample_col(df_rdp_rel_f) %>%
  select(any_of(c("SampleID", paste0("f_", family_targets)))) %>%
  rename_with(~ gsub("^f[_.: ]+", "", ., ignore.case = TRUE), .cols = -SampleID) %>%
  pivot_longer(-SampleID, names_to = "Family", values_to = "abundance") %>%
  mutate(abundance = to_num(abundance)) %>%
  filter(Family %in% family_targets, is.finite(abundance)) %>%
  transmute(SampleID, Family, abundance)

silva_fam_long <- add_sample_col(df_silva_rel_f) %>%
  select(any_of(c("SampleID", paste0("f_", family_targets)))) %>%
  rename_with(~ gsub("^f[_.: ]+", "", ., ignore.case = TRUE), .cols = -SampleID) %>%
  pivot_longer(-SampleID, names_to = "Family", values_to = "abundance") %>%
  mutate(abundance = to_num(abundance)) %>%
  filter(Family %in% family_targets, is.finite(abundance)) %>%
  transmute(SampleID, Family, abundance)

# --- Спирман по выбранным СЕМЕЙСТВАМ (без каких-либо суммирований) ---
spearman_by_family_simple <- function(pcr_fam, ngs_fam, label) {
  out <- inner_join(pcr_fam, ngs_fam, by = c("SampleID","Family"),
                    suffix = c("_PCR","_NGS")) %>%
    group_by(Family) %>%
    summarize(
      Genus = paste0(first(Family), " (family)"),   # для общей оси Y
      n     = n(),
      rho   = if (n() >= min_n_pairs &&
                  sd(abundance_PCR) > 0 && sd(abundance_NGS) > 0)
        suppressWarnings(cor(abundance_PCR, abundance_NGS, method = "spearman"))
      else NA_real_,
      pval  = if (n() >= min_n_pairs &&
                  sd(abundance_PCR) > 0 && sd(abundance_NGS) > 0)
        tryCatch(suppressWarnings(
          cor.test(abundance_PCR, abundance_NGS, method = "spearman")$p.value),
          error = function(e) NA_real_)
      else NA_real_,
      .groups = "drop"
    ) %>%
    ungroup() %>%
    mutate(
      qval = p.adjust(pval, method = "BH"),
      comparison = label
    )
  
  out
}

res_fam_rdp   <- spearman_by_family_simple(pcr_fam_long, rdp_fam_long,   "PCR vs RDP")
res_fam_silva <- spearman_by_family_simple(pcr_fam_long, silva_fam_long, "PCR vs SILVA")
res_fam_rdp_silva<- spearman_by_family_simple(rdp_fam_long, silva_fam_long, "RDP vs SILVA")
# --- Спирман по РОДАМ (как у вас было; НИЧЕГО не исключаем) ---
res_rdp_genus   <- spearman_by_genus(pcr_long,   rdp_long,   "PCR vs RDP")
res_silva_genus <- spearman_by_genus(pcr_long,   silva_long, "PCR vs SILVA")

# --- Объединяем генусы + выбранные семейства и дальше строим график как раньше ---
res_all <- bind_rows(res_rdp_genus, res_silva_genus, res_fam_rdp, res_fam_silva) %>%
  mutate(
    rho_clr = ifelse(is.na(rho), 0.001, pmin(pmax(rho, 0), 1)),
    size_num = case_when(
      is.na(pval)  ~ 3,
      pval < 0.001 ~ 25,
      pval < 0.01  ~ 18,
      pval < 0.05  ~ 12,
      TRUE         ~ 9
    )
  )
print(res_all)

#boxplot 



is_family        <- grepl("aceae$", targets_display, ignore.case = TRUE)
target_families  <- targets_display[is_family]
target_genera    <- sub("\\.$","", sub("\\s+.*$", "", targets_display[!is_family]))

display_map <- c(
  setNames(target_families, tolower(target_families)),
  setNames(targets_display[!is_family], tolower(target_genera))
)

## ===== Хелперы =====
to_num <- function(x) suppressWarnings(as.numeric(gsub(",", ".", x)))

ensure_sample_id <- function(df){
  df <- as.data.frame(df)
  idc <- intersect(names(df), c("SampleID","Sample","Name","ID","Run"))
  if (length(idc)) dplyr::rename(df, SampleID = !!idc[1]) else tibble::rownames_to_column(df, "SampleID")
}

scale_to_fraction <- function(x){
  x <- to_num(x)
  if (!any(is.finite(x))) return(x)
  if (max(x, na.rm = TRUE) > 1.5) x/100 else x
}

clean_taxon <- function(nm){
  nm %>%
    gsub("^f[_.: ]+", "", ., ignore.case = TRUE) %>%
    gsub("^g(__|[_.: ]+)", "", ., ignore.case = TRUE)
}

recode_ruminococcus <- function(x){
  dplyr::case_when(
    !grepl("aceae$", x, ignore.case = TRUE) & grepl("(?i)^ruminococc", x) ~ "Ruminococcus",
    !grepl("aceae$", x, ignore.case = TRUE) & grepl("(?i)ruminococc.*group", x) ~ "Ruminococcus",
    !grepl("aceae$", x, ignore.case = TRUE) & grepl("(?i)ruminococc.*ucg",   x) ~ "Ruminococcus",
    !grepl("aceae$", x, ignore.case = TRUE) & grepl("(?i)ruminococcaid|ruminococcid", x) ~ "Ruminococcus",
    TRUE ~ x
  )
}

## ===== PCR: wide -> long (починенная) =====
to_long_pcr <- function(df_pcr){
  dfp <- ensure_sample_id(df_pcr)
  taxa_cols <- grep("^PCR_", names(dfp), value = TRUE)
  taxa_cols <- taxa_cols[!grepl("_?Total$", taxa_cols, ignore.case = TRUE)]
  if (!length(taxa_cols)) stop("В df_pcr нет колонок PCR_*")
  
  dfp %>%
    select(SampleID, all_of(taxa_cols)) %>%
    mutate(across(-SampleID, to_num)) %>%
    pivot_longer(
      -SampleID,
      names_to  = "tax_col",    # не 'col'!
      values_to = "abundance"
    ) %>%
    mutate(
      raw_taxon = gsub("^PCR_", "", .data$tax_col),
      raw_taxon = trimws(gsub("\\s*\\(.*\\)$", "", raw_taxon)),   # убрать хвосты в скобках
      raw_taxon = gsub("\\s+", " ", raw_taxon),                   # схлопнуть пробелы
      is_fam    = grepl("aceae\\b", raw_taxon, ignore.case = TRUE),
      key = ifelse(
        is_fam,
        trimws(sub("\\.+$", "", raw_taxon)),                      # family как есть
        trimws(gsub("\\.$","", sub("\\s+.*$", "", raw_taxon)))    # genus = первое слово, без точки
      )
    ) %>%
    group_by(SampleID, key) %>%
    summarise(abundance = sum(abundance, na.rm = TRUE), .groups = "drop") %>%
    mutate(Source = "PCR")
}

## ===== NGS относительные -> «квази-абсолюты» (домножение на PCR_Total) =====
ngs_rel_to_abs <- function(df_rel, source_label, recode_rum = TRUE){
  dfn <- ensure_sample_id(df_rel)
  tax_cols <- setdiff(names(dfn), "SampleID")
  
  dfn %>%
    mutate(across(all_of(tax_cols), scale_to_fraction)) %>%
    left_join(
      ensure_sample_id(df_pcr) %>% transmute(SampleID, PCR_Total = to_num(PCR_Total)),
      by = "SampleID"
    ) %>%
    mutate(across(all_of(tax_cols), ~ .x * PCR_Total)) %>%
    pivot_longer(all_of(tax_cols), names_to = "Taxon_raw", values_to = "abundance") %>%
    mutate(
      key = clean_taxon(Taxon_raw),
      key = if (recode_rum) recode_ruminococcus(key) else key,
      Source = paste0(source_label, "×PCR_Total")
    ) %>%
    select(SampleID, key, abundance, Source)
}

## ===== Построение данных =====
# PCR long
# Нормализуем ключи для PCR: род = первое слово (учитывая подчёркивания)
pcr_long <- to_long_pcr(df_pcr)
pcr_long <- pcr_long %>%
  mutate(
    key = ifelse(
      grepl("aceae$", key, ignore.case = TRUE),
      key,  # семейства не трогаем
      {
        k <- gsub("_", " ", key)
        k <- trimws(gsub("\\.$", "", sub("\\s+.*$", "", k)))
        k
      }
    )
  )


# NGS генусы: из df_rdp_rel / df_silva_rel с перекодировкой Ruminococc*
rdp_abs_gen   <- ngs_rel_to_abs(df_rdp_rel,   "RDP",   recode_rum = FALSE)
silva_abs_gen <- ngs_rel_to_abs(df_silva_rel, "SILVA", recode_rum = FALSE)

# NGS семейства: из df_rdp_rel_f / df_silva_rel_f (без перекодировки)
rdp_abs_fam   <- ngs_rel_to_abs(df_rdp_rel_f,   "RDP",   recode_rum = FALSE)
silva_abs_fam <- ngs_rel_to_abs(df_silva_rel_f, "SILVA", recode_rum = FALSE)

## ===== Фильтрация: ровно нужные таксоны =====
pick_genera   <- tolower(target_genera)
pick_families <- tolower(target_families)

pcr_sel <- pcr_long %>%
  filter(tolower(key) %in% c(pick_genera, pick_families))

rdp_sel <- bind_rows(
  rdp_abs_gen %>% filter(tolower(key) %in% pick_genera),
  rdp_abs_fam %>% filter(tolower(key) %in% pick_families)
)

silva_sel <- bind_rows(
  silva_abs_gen %>% filter(tolower(key) %in% pick_genera),
  silva_abs_fam %>% filter(tolower(key) %in% pick_families)
)

## ===== Сборка и подписи для оси X =====
add_display <- function(df){
  df %>%
    mutate(Display = {
      disp <- unname(display_map[tolower(key)])
      ifelse(is.na(disp) | disp == "", key, disp)
    })
}

plot_df <- bind_rows(
  add_display(pcr_sel),
  add_display(rdp_sel),
  add_display(silva_sel)
)

# 0) безопасные данные для оси Y
plot_df2 <- plot_df %>%
  mutate(y = pmax(abundance, 0)) %>%       # всё <0 -> 0
  filter(is.finite(y) | is.na(y))

# фиксируем 11 таксонов на оси X
plot_df2$Display <- factor(plot_df2$Display, levels = targets_display)

# верхний предел (чтобы не было max==min/NA)
y_cap <- suppressWarnings(quantile(plot_df2$y, probs = 0.995, na.rm = TRUE))
y_cap <- as.numeric(y_cap)
if (!is.finite(y_cap) & y_cap <= 0) {
  y_cap <- max(plot_df2$y, na.rm = TRUE)
  if (!is.finite(y_cap) &  y_cap <= 0) y_cap <- 1
}

# ——— большой график (лог1p, узкие боксы, без «пляшущей» сетки) ———
p_big <- ggplot(plot_df2, aes(x = Display, y = y, fill = Source)) +
  #geom_violin(scale = 'width', position = 'dodge', trim = F, alpha=0.4)+
  geom_boxplot(
    width = 0.42,
    position = position_dodge2(width = 0.45, padding = 0, preserve = "single"),
    outlier.alpha = 0.35) + 
  scale_fill_manual(values = c("PCR"="#2E86AB","RDP×PCR_Total"="#3DA35D","SILVA×PCR_Total"="#E07A5F")) +
  scale_x_discrete(drop = FALSE) +
  scale_y_continuous(
    trans  = "log1p",
    limits = c(0, 9000000), 
    breaks = c(0,1,10,100,1000,10000,1e5,1e6,1e7),# безопасные границы
    oob    = scales::oob_squish        # не удалять выбросы, а «прижимать»
    # не задаём breaks вручную — это и вызывало NaN
  ) +
  labs(x = NULL, y = "Absolute abundance (log1p)")+
  #title = "Absolute PCR vs NGS (relative × PCR_Total): selected taxa") +
  guides(fill = guide_legend(nrow = 1)) +
  theme_bw(base_size = 20) +
  theme(
    legend.justification = c("left","top"),
    legend.position    = "top",
    legend.margin      = margin(b = 2),
    legend.text        = element_text(size=16),
    legend.title       = element_text(size=16),
    axis.text.x        = element_text(angle = 30, hjust = 1, vjust = 1),
    panel.grid.minor   = element_blank(),
    panel.grid.major.x = element_blank(),
    plot.title         = element_text(margin = margin(b = 6)),
    plot.margin        = margin(t = 4, r = 6, b = 6, l = 6),
    aspect.ratio       = 0.50
  )

# ——— маленький график (линейная шкала, белый фон, без подписей) ———
top_outs <- plot_df2[order(plot_df2$y, decreasing = T)[1:4],]
p_small <- ggplot(plot_df2, aes(x = Display, y = y, fill = Source)) +
  geom_boxplot(
    width = 0.7,
    position = position_dodge2(width = 0.28, padding = 0, preserve = "single"),
    outlier.alpha = 0.7, outlier.size = 0.85
  ) +
  scale_fill_manual(values = c("PCR"="#2E86AB","RDP×PCR_Total"="#3DA35D","SILVA×PCR_Total"="#E07A5F")) +
  scale_color_manual(values = c("PCR"="#2E86AB","RDP×PCR_Total"="#3DA35D","SILVA×PCR_Total"="#E07A5F")) +
  scale_x_discrete(drop = FALSE, labels=c('Bt','Bf','Ch','En','Os','Fp','Lc',
                                          'Od','Ru','Ef','Su')) +
  xlab('')+ylab('')+
  coord_cartesian(ylim = c(0, y_cap)) +
  theme_light(base_size = 15) +
  theme(
    panel.background = element_rect(fill = 'white', colour = NA),
    plot.background  = element_rect(fill = NA, colour = NA),
    legend.position  = "none",
    #    axis.title       = element_blank(),
    #    axis.text        = element_blank(),
    #    axis.ticks       = element_blank(),
    #    panel.grid       = element_blank(),
    plot.margin      = margin(0,0,0,0)
  )

# ——— вставка в правый верхний угол панели, не задевая легенду ———
p_final <- p_big +
  inset_element(
    p_small,
    left = 0.63, right = 0.999,
    bottom = 0.69, top = 0.999,
    align_to = "panel"
  )

print(p_final)

# компактное сохранение без лишнего воздуха
ggsave("boxplots_with_inset.png", p_final, width = 12, height = 10, dpi = 300)

