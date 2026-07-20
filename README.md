# topographic_alignment
Code for Brain Hierarchy as Coordinated Regional Structure-Function Topographic Alignment.


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
