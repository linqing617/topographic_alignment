```r
# =============================================================================
# SCRIPT: GO Biological Process Enrichment Analysis
# =============================================================================
#
# PURPOSE:
# Perform GO Biological Process (BP) enrichment analyses for positive and
# negative TAN-associated gene sets separately.
#
# INPUT:
#   - Positive genes: s400_TAN_RSN_posoverlap(z=5).csv
#   - Negative genes: s400_TAN_RSN_negoverlap(z=5).csv
#
# Each input CSV must contain a column named "gene" with human gene symbols.
#
# REQUIREMENTS:
#   - clusterProfiler
#   - org.Hs.eg.db
#   - ggplot2
#
# OUTPUT:
#   - Gene-to-Entrez ID mapping tables
#   - Unmapped gene lists
#   - GO-BP enrichment result tables
#   - GO-BP dot plots (PDF)
#   - Summary table
#
# NOTE:
#   - Gene symbols are converted to Entrez Gene IDs using org.Hs.eg.db.
#   - Multiple-testing correction uses the Benjamini-Hochberg method.
#   - The default enrichment background is used unless a custom universe
#     is explicitly supplied.
#
# =============================================================================


# 1. Setup --------------------------------------------------------------------

rm(list = ls())

library(clusterProfiler)
library(org.Hs.eg.db)
library(ggplot2)


# 2. Configure paths and parameters -------------------------------------------

# Run this script from the repository root directory.
input_dir <- file.path("data", "gene")
output_dir <- file.path("results", "gene_enrichment")

positive_file <- file.path(
  input_dir,
  "s400_TAN_RSN_posoverlap(z=5).csv"
)

negative_file <- file.path(
  input_dir,
  "s400_TAN_RSN_negoverlap(z=5).csv"
)

dir.create(
  output_dir,
  recursive = TRUE,
  showWarnings = FALSE
)

pvalue_cutoff <- 0.05
qvalue_cutoff <- 0.05
p_adjust_method <- "BH"
ontology <- "BP"


# 3. Read and validate gene lists ---------------------------------------------

read_gene_list <- function(file) {
  
  if (!file.exists(file)) {
    stop("Input file does not exist: ", file)
  }
  
  dat <- read.csv(
    file,
    stringsAsFactors = FALSE
  )
  
  if (!"gene" %in% colnames(dat)) {
    stop(
      "Input file must contain a column named 'gene': ",
      file
    )
  }
  
  genes <- trimws(as.character(dat$gene))
  genes <- genes[!is.na(genes) & genes != ""]
  genes <- unique(genes)
  
  if (length(genes) == 0) {
    stop("No valid gene symbols found in: ", file)
  }
  
  message(
    basename(file), ": ",
    length(genes), " unique gene symbols loaded."
  )
  
  return(genes)
}


posgene <- read_gene_list(positive_file)
neggene <- read_gene_list(negative_file)


# 4. Convert gene symbols to Entrez IDs ---------------------------------------

map_gene_ids <- function(genes, label, output_dir) {
  
  mapped <- bitr(
    genes,
    fromType = "SYMBOL",
    toType = "ENTREZID",
    OrgDb = org.Hs.eg.db
  )
  
  mapped <- unique(mapped)
  
  mapped_symbols <- unique(mapped$SYMBOL)
  unmapped_genes <- setdiff(genes, mapped_symbols)
  
  # Save the gene mapping table
  write.csv(
    mapped,
    file = file.path(
      output_dir,
      paste0(label, "_gene_mapping.csv")
    ),
    row.names = FALSE
  )
  
  # Save unmapped gene symbols
  write.csv(
    data.frame(gene = unmapped_genes),
    file = file.path(
      output_dir,
      paste0(label, "_unmapped_genes.csv")
    ),
    row.names = FALSE
  )
  
  message(
    label, ": ",
    length(mapped_symbols), "/", length(genes),
    " input symbols mapped to Entrez IDs; ",
    length(unmapped_genes), " symbols not mapped."
  )
  
  if (nrow(mapped) == 0) {
    stop(
      "No genes could be mapped for gene set: ",
      label
    )
  }
  
  # Remove duplicated Entrez IDs
  mapped <- mapped[
    !duplicated(mapped$ENTREZID),
    ,
    drop = FALSE
  ]
  
  return(mapped)
}


pos <- map_gene_ids(
  posgene,
  "positive",
  output_dir
)

neg <- map_gene_ids(
  neggene,
  "negative",
  output_dir
)


# 5. GO-BP enrichment analysis ------------------------------------------------

run_go_enrichment <- function(mapped_genes, label, output_dir) {
  
  ego <- enrichGO(
    gene = unique(mapped_genes$ENTREZID),
    keyType = "ENTREZID",
    OrgDb = org.Hs.eg.db,
    ont = ontology,
    pvalueCutoff = pvalue_cutoff,
    pAdjustMethod = p_adjust_method,
    qvalueCutoff = qvalue_cutoff,
    readable = TRUE
  )
  
  # Export enrichment results
  result_df <- as.data.frame(ego)
  
  write.csv(
    result_df,
    file = file.path(
      output_dir,
      paste0(label, "_GO_BP_enrichment.csv")
    ),
    row.names = FALSE
  )
  
  # Generate and save dot plot if enriched terms are available
  if (nrow(result_df) > 0) {
    
    show_n <- if (label == "positive") 15 else 10
    
    p <- dotplot(
      ego,
      x = "GeneRatio",
      color = "p.adjust",
      showCategory = show_n
    ) +
      ggtitle(
        if (label == "positive") {
          "Positive TAN-associated genes"
        } else {
          "Negative TAN-associated genes"
        }
      ) +
      theme_classic()
    
    ggsave(
      filename = file.path(
        output_dir,
        paste0(label, "_GO_BP_dotplot.pdf")
      ),
      plot = p,
      width = 9,
      height = 7
    )
    
  } else {
    
    message(
      "No GO-BP terms passed the specified cutoffs for: ",
      label
    )
  }
  
  return(ego)
}


# 6. Run enrichment for positive and negative genes ---------------------------

pos_BP <- run_go_enrichment(
  pos,
  "positive",
  output_dir
)

neg_BP <- run_go_enrichment(
  neg,
  "negative",
  output_dir
)


# 7. Export summary ------------------------------------------------------------

summary_df <- data.frame(
  GeneSet = c("Positive", "Negative"),
  InputGenes = c(
    length(posgene),
    length(neggene)
  ),
  MappedSymbols = c(
    length(unique(pos$SYMBOL)),
    length(unique(neg$SYMBOL))
  ),
  UniqueEntrezIDs = c(
    length(unique(pos$ENTREZID)),
    length(unique(neg$ENTREZID))
  ),
  EnrichedTerms = c(
    nrow(as.data.frame(pos_BP)),
    nrow(as.data.frame(neg_BP))
  )
)

write.csv(
  summary_df,
  file = file.path(
    output_dir,
    "GO_BP_summary.csv"
  ),
  row.names = FALSE
)

print(summary_df)

message("TAN GO-BP enrichment analyses completed.")
message("Results saved to: ", output_dir)
```