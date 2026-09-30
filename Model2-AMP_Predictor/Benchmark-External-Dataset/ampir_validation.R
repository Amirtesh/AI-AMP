#!/usr/bin/env Rscript
if (!requireNamespace("ampir", quietly = TRUE)) install.packages("ampir", repos = "https://cloud.r-project.org")
if (!requireNamespace("pROC", quietly = TRUE)) install.packages("pROC", repos = "https://cloud.r-project.org")
if (!requireNamespace("PRROC", quietly = TRUE)) install.packages("PRROC", repos = "https://cloud.r-project.org")

library(ampir); library(pROC); library(PRROC)

df <- read.csv("Cleaned_External_Dataset.csv", stringsAsFactors = FALSE)
input_df <- data.frame(seq_name = as.character(df$id_ref), seq_aa = df$Sequence, stringsAsFactors = FALSE)

pred <- predict_amps(input_df, model = "mature")

# --- sanity checks before trusting anything downstream ---
cat("Rows in input:", nrow(input_df), " Rows in predictions:", nrow(pred), "\n")
cat("NA count in prob_AMP:", sum(is.na(pred$prob_AMP)), "\n")
cat("prob_AMP range:", range(pred$prob_AMP, na.rm = TRUE), "\n")
print(head(pred[, c("seq_name", "prob_AMP")], 10))

df$prob_AMP <- pred$prob_AMP
df$Label <- as.numeric(df$Label)

roc_obj <- roc(df$Label, df$prob_AMP, quiet = TRUE)
auroc <- as.numeric(auc(roc_obj))
pr <- pr.curve(scores.class0 = df$prob_AMP[df$Label == 1],
                scores.class1 = df$prob_AMP[df$Label == 0], curve = FALSE)
auprc <- pr$auc.integral

threshold <- 0.5
pc <- ifelse(df$prob_AMP >= threshold, 1, 0)
tp <- sum(pc == 1 & df$Label == 1); fp <- sum(pc == 1 & df$Label == 0)
tn <- sum(pc == 0 & df$Label == 0); fn <- sum(pc == 0 & df$Label == 1)
precision <- tp / (tp + fp); recall <- tp / (tp + fn)
f1 <- 2 * precision * recall / (precision + recall)

# fixed: cast to double before multiplying, avoids integer overflow -> NA
mcc <- (as.double(tp) * as.double(tn) - as.double(fp) * as.double(fn)) /
       sqrt(as.double(tp+fp) * as.double(tp+fn) * as.double(tn+fp) * as.double(tn+fn))

cat(sprintf("\nthreshold,AUROC,AUPRC,MCC,F1,precision,recall\n%.3f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
            threshold, auroc, auprc, mcc, f1, precision, recall))
write.csv(df, "ampir_predictions.csv", row.names = FALSE)
