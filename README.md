#  Brain Hierarchy as Coordinated Regional Structure-Function Topographic Alignment.
This repository contains code and data created in support of my project "Brain hierarchy as coordinated regional structure-function topographic alignment".

## Overview

By leveraging five independent structural MRI cohorts spanning the lifespan 
(HCP-YA, UK Biobank, ZIB, ABIDE, and ADNI; combined N > 39,000, ages 6–95 years), 
we developed a **Topographic Alignment (TA) framework** to characterize the 
relationship between regional gray matter organization and cognitive functional 
representations.

The framework consists of two major components:

1. **TA score profile**
   
   For each individual, we calculate the cosine similarity between regional 
   voxel-wise gray matter volume (GMV) topography and 539 Neurosynth-derived 
   meta-analytic cognitive activation topographies. This generates an 
   individual-level TA profile representing the alignment between brain structure 
   and cognitive functions.

2. **Topographic Alignment Network (TAN)**

   We construct a network based on the inter-regional similarity of TA score 
   profiles, capturing large-scale organizational principles of structure–function 
   alignment across the human cortex.

## Code
The code folder contains main code used to run the analyses . If you use this code, don't forget to change the path variable
- **N0_make_network:** Compute the individual-level TA score profile and TA network. 
This module calls the function `topographic_alignment.m`.
- **N1_term_freq:** Perform term frequency analysis to investigate the relationship 
  between positive TA scores, negative TA scores, and cognitive functional 
  representations.This module calls the function `extractTerms.m` and `word Frequeny.m`.
- **N2_sa:** Calculate the gradient of the topographic alignment network (TAN) and 
compare it with the sensorimotor–association axis（S-A rank） to investigate the hierarchical organization of TAN.
- **N3_gene:** Investigate the shared biological basis underlying the topographic alignment network (TAN) and resting-state functional connectivity (RSFC) by performing PLS analysis and gene ontology (GO) enrichment analysis.
- **N4_SFC:** Investigate the contribution of the topographic alignment tetwork (TAN) to enhanced structure–function coupling by applying a structural–functional coupling model.









