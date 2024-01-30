# AGNOSTIC: Adaptable Generalized Neural-Network Open-source Spectroscopy Training dataset of Individual Components
<div>
Published in Imaging Neuroscience: <a href="https://doi.org/10.1162/imag_a_00025">Application of a 1H brain MRS benchmark dataset to deep learning for out-of-voxel artifacts</a>
</div>
---

<ul style="list-style: none;">
  <li><b>Aaron T. Gudmundson</b>, Johns Hopkins School of Medicine, Kennedy Krieger Institute, <a href="https://orcid.org/0000-0001-5104-0959">ORCID: 0000-0001-5104-0959</a></li>
  <li><b>Christopher W. Davies-Jenkins</b>, Johns Hopkins School of Medicine, Kennedy Krieger Institute, <a href="https://orcid.org/0000-0002-6015-762X">ORCID: 0000-0002-6015-762X</a></li>
  <li><b>İpek Özdemir</b>, Johns Hopkins School of Medicine, Kennedy Krieger Institute, <a href="https://orcid.org/0000-0001-6807-9390">ORCID: 0000-0001-6807-9390</a></li>
  <li><b>Saipavitra Murali-Manohar</b>, Johns Hopkins School of Medicine, Kennedy Krieger Institute, <a href="https://orcid.org/0000-0002-4978-0736">ORCID: 0000-0002-4978-0736</a></li>
  <li><b>Helge J. Zöllner</b>, Johns Hopkins School of Medicine, Kennedy Krieger Institute, <a href="https://orcid.org/0000-0002-7148-292X">ORCID: 0000-0002-7148-292X</a></li>
  <li><b>Yulu Song</b>, Johns Hopkins School of Medicine, Kennedy Krieger Institute, <a href="https://orcid.org/0000-0002-4416-7959">ORCID: 0000-0002-4416-7959</a></li>
  <li><b>Kathleen E. Hupfeld</b>, Johns Hopkins School of Medicine, Kennedy Krieger Institute, <a href="https://orcid.org/0000-0001-5086-4841">ORCID: 0000-0001-5086-4841</a></li>
  <li><b>Alfons Schnitzler</b>, Heinrich-Heine-University Düsseldorf, <a href="https://orcid.org/0000-0002-6414-7939">ORCID: 0000-0002-6414-7939</a></li>
  <li><b>Georg Oeltzschner</b>, Johns Hopkins School of Medicine, Kennedy Krieger Institute, <a href="https://orcid.org/0000-0003-3083-9811">ORCID: 0000-0003-3083-9811</a></li>
  <li><b>Craig E. L. Stark</b>, University of California, Irvine, <a href="https://orcid.org/0000-0002-9334-8502">ORCID: 0000-0002-9334-8502</a></li>
  <li><b>Richard A. E. Edden</b>, Johns Hopkins School of Medicine, Kennedy Krieger Institute, <a href="https://orcid.org/0000-0002-0671-7374">ORCID: 0000-0002-0671-7374</a></li>
</ul>

## Publication Information Citation
> Gudmundson, A. T., Davies-Jenkins, C. W., Özdemir, İ., Murali-Manohar, S., Zöllner, H. J., Song, Y., ... & Edden, R. A. (2023). Application of a 1H brain MRS benchmark dataset to deep learning for out-of-voxel artifacts. Imaging Neuroscience, 1, 1-15. https://doi.org/10.1162/imag_a_00025


## Dataset Description
The Adaptable Generalized Neural-Network Open-source Spectroscopy Training dataset of Individual Components (AGNOSTIC), is a dataset consisting of 259,200 synthetic MRS examples. The synthetic examples contained within the dataset were produced to resemble in vivo brain data with metabolite, macromolecule, residual water signals, and noise. The parameter space that AGNOSTIC spans is wide-reaching, comprising: 18 field strengths; 15 echo times; broad distributions of metabolite, MM, and water amplitudes; and densely sampled time-domain to allow down-sampling. 

## Abstract
Neural networks are potentially valuable for many of the challenges associated with MRS data. The purpose of this manuscript is to describe the AGNOSTIC dataset, which contains 259,200 synthetic 1H MRS examples for training and testing neural networks. AGNOSTIC was created using 270 basis sets that were simulated across 18 field strengths and 15 echo times. The synthetic examples were produced to resemble in vivo brain data with combinations of metabolite, macromolecule, residual water signals, and noise. To demonstrate the utility, we apply AGNOSTIC to train two Convolutional Neural Networks (CNNs) to address out-of-voxel (OOV) echoes. A Detection Network was trained to identify the point-wise presence of OOV echoes, providing proof of concept for real-time detection. A Prediction Network was trained to reconstruct OOV echoes, allowing subtraction during post-processing. Complex OOV signals were mixed into 85% of synthetic examples to train two separate CNNs for the detection and prediction of OOV signals. AGNOSTIC is available through Dryad, and all Python 3 code is available through GitHub. The Detection network was shown to perform well, identifying 95% of OOV echoes. Traditional modeling of these detected OOV signals was evaluated and may prove to be an effective method during linear-combination modeling. The Prediction Network greatly reduces OOV echoes within FIDs and achieved a median log10 normed-MSE of—1.79, an improvement of almost two orders of magnitude.

## Description of the Data and File Structure
The dataset is structured as a zipped NumPy archive file (.npz). The zipped NumPy archive file contains complex-valued NumPy arrays of time-domain (4096 timepoints) data corresponding to the metabolite, macromolecule, water, and noise components which can be combined in different ways depending on the users goal or objective. Within the file, all the acquisition parameters (field strength, echo time, spectral width, etc.), simulation parameters (signal to noise, full-width half-max, concentrations, T2 relaxation, etc.), and data augmentation options are specified. 


## Sharing/Access Information
Dataset can be found at Dryad:
  * https://datadryad.org/stash/dataset/doi:10.7280/D1RX1T
Data was derived from the following sources:
  * https://github.com/agudmundson/agnostic

## Code/Software
The following Python 3 scripts, found at https://github.com/agudmundson/agnostic, were used to generate AGNOSTIC:
  * 00_simulation.py (Density Matrix Simulation Functions)
  * 01_deep_sim.py   (Acquisition Setting and Metabolite Simulations)
  * 02_metab_matrix_py (Restructuring and Normalizing Basis Set)
  * 03_gen_data.py   (Synthetic Dataset)
  * 04_randomize.py  (Randomizing Field Strengths and Echo Times)

These scripts primarily rely upon NumPy, SciPy, and standard built-in Python libaries (os, glob, subprocess, etc.)

## Dataset Contains:
<table>
  <tr>
    <th>Column Name   </th>
    <th>Datatype      </th>
    <th>Shape         </th>
    <th>Description   </th>
  </tr>
  <tr>
    <td>Dataset       </td>
    <td>String        </td>
    <td>1             </td>
    <td>Dataset used in Data Simulations for Concentrations</td>
  </tr>
  <tr>
    <td>Field_Str     </td>
    <td>Array         </td>
    <td>Batch x 1     </td>
    <td>Field Strength Used (Tesla)</td>
  </tr>
  <tr>
    <td>Echo_Times    </td>
    <td>Array         </td>
    <td>Batch x 1     </td>
    <td>Echo Times Used (ms)</td>
  </tr>
  <tr>
    <td>sw            </td>
    <td>Array         </td>
    <td>Batch x 1     </td>
    <td>Spectral Widths Available (Hz) (Dependent on Subsampling and Field Strength)</td>
  </tr>
  <tr>
    <td>subsample     </td>
    <td>Array         </td>
    <td>Batch x 1     </td>
    <td>Subsampling Stride Corresponding to SpecWidth</td>
  </tr>
  <tr>
    <td>nPoints       </td>
    <td>Array         </td>
    <td>Batch x 1     </td>
    <td>Number of Points (i.e. 512,1024,2048)</td>
  </tr>
  <tr>
    <td>Metab         </td>
    <td>Array         </td>
    <td>Batch x 4096  </td>
    <td>Full Metabolite Signal (w/ Concentration, Lorentzian LB, and Gaussian LB)</td>
  </tr>
  <tr>
    <td>MM            </td>
    <td>Array         </td>
    <td>Batch x 4096  </td>
    <td>Full Macromolecule Signal (w/ Concentration, Lorentzian LB, and Gaussian LB)</td>
  </tr>
  <tr>
    <td>water         </td>
    <td>Array         </td>
    <td>Batch x 4096  </td>
    <td>Full Water Signal (w/ <=5Components, Lorentzian LB, Gaussian LB, Scaling 5x-20x Metabolites)</td>
  </tr>
  <tr>
    <td>noise         </td>
    <td>Array         </td>
    <td>Batch x 4096  </td>
    <td>Normal Distributed Noise</td>
  </tr>
  <tr>
    <td>time          </td>
    <td>Array         </td>
    <td>Batch x 4096  </td>
    <td>Time Axis (seconds)</td>
  </tr>
  <tr>
    <td>ppm           </td>
    <td>Array         </td>
    <td>Batch x 1     </td>
    <td>Frequency Axis (ppm)</td>
  </tr>
  <tr>
    <td>Amplitude     </td>
    <td>Array         </td>
    <td>Batch x 182   </td>
    <td>Concentration Used for Each Spin (Metabolite & MM)</td>
  </tr>
  <tr>
    <td>water_pos     </td>
    <td>Array         </td>
    <td>Batch x 1     </td>
    <td>Water is Positive or Negative (0=Pos; 1=Neg)</td>
  </tr>
  <tr>
    <td>water_comp    </td>
    <td>Array         </td>
    <td>Batch x 5     </td>
    <td>PPM value of Each Component</td>
  </tr>
  <tr>
    <td>waterNcomp    </td>
    <td>Array         </td>
    <td>Batch x 5     </td>
    <td>Components Included</td>
  </tr>
  <tr>
    <td>water_amp     </td>
    <td>Array         </td>
    <td>Batch x 5     </td>
    <td>Water Scaling </td>
  </tr>
  <tr>
    <td>noise_amp     </td>
    <td>Array         </td>
    <td>Batch x 1     </td>
    <td>Noise Scaling (Equivalent SNR Range Across SpecWidth and Npoints)</td>
  </tr>
  <tr>
    <td>freq_shift    </td>
    <td>Array         </td>
    <td>Batch x 1     </td>
    <td>Frequency Shifts</td>
  </tr>
  <tr>
    <td>phase0        </td>
    <td>Array         </td>
    <td>Batch x 1     </td>
    <td>0th Order Phase</td>
  </tr>
  <tr>
    <td>phase1        </td>
    <td>Array         </td>
    <td>Batch x 1     </td>
    <td>1st Order Phase</td>
  </tr>
  <tr>
    <td>phase1_piv    </td>
    <td>Array         </td>
    <td>Batch x 1     </td>
    <td>1st Order Phase Pivot Point</td>
  </tr>
  <tr>
    <td>SNR           </td>
    <td>Array         </td>
    <td>Batch x 1     </td>
    <td>SNR (NAA_Amp / StdDev_Noise)</td>
  </tr>
  <tr>
    <td>LBL           </td>
    <td>Array         </td>
    <td>Batch x 182   </td>
    <td>Lorentzian Line Broadening Metab/MM</td>
  </tr>
  <tr>
    <td>LBG           </td>
    <td>Array         </td>
    <td>Batch x 182   </td>
    <td>Gaussian Line Broadening Metab/MM</td>
  </tr>
  <tr>
    <td>m_mult        </td>
    <td>Array         </td>
    <td>Batch x 2     </td>
    <td>Norm Metab (Metab --> 1)</td>
  </tr>
  <tr>
    <td>w_mult        </td>
    <td>Array         </td>
    <td>Batch x 2     </td>
    <td>Norm Water (Water --> 5x-20x) Correctly Scales Water Relative to Metab</td>
  </tr>
  <tr>
    <td>LBL_Water     </td>
    <td>Array         </td>
    <td>Batch x 5     </td>
    <td>Lorentzian Line Broadening Water</td>
  </tr>
  <tr>
    <td>LBG_Water     </td>
    <td>Array         </td>
    <td>Batch x 5     </td>
    <td>Gaussian Line Broadening Water</td>
  </tr>
  <tr>
    <td>FWHM_MM       </td>
    <td>Array         </td>
    <td>Batch x 14    </td>
    <td>Target FWHM of Macromolecules (14 Macromolecules)</td>
  </tr>
  <tr>
    <td>FWHM_Metab    </td>
    <td>Array         </td>
    <td>Batch x 14    </td>
    <td>Target FWHM of Metaboites (FWHM of NAA)</td>
  </tr>
  <tr>
    <td>Healthy       </td>
    <td>Array         </td>
    <td>Batch x 1     </td>
    <td>Healthy = 0; Clinical = 1</td>
  </tr>
  <tr>
    <td>Clinical      </td>
    <td>Array         </td>
    <td>Batch x 1     </td>
    <td>Healthy = 0; Clinical > 0 (See Clin_Names)</td>
  </tr>
  <tr>
    <td>Clin_Names    </td>
    <td>List          </td>
    <td>21            </td>
    <td>Corresponding Names of Population Number from 'Clinical' | Note* 0 is Healthy</td>
  </tr>
  <tr>
    <td>Drop_Sig      </td>
    <td>Array         </td>
    <td>Batch x 4096  </td>
    <td>Some/All Metab/MM Signal to be <b>Subtracted</b> (See Batch_Drop and dIdx_Drop)</td>
  </tr>
  <tr>
    <td>Batch_Drop    </td>
    <td>Array         </td>
    <td>Batch x 1     </td>
    <td>Randomly Leave Off Some/All Metabolites/Macromolecules - Indicates Which Index was Selected</td>
  </tr>
  <tr>
    <td>dIdx_Drop     </td>
    <td>Array         </td>
    <td>Batch x 1     </td>
    <td>Randomly Leave Off Some/All Metabolites/Macromolecules - Index of the 182 Spins to Drop</td>
  </tr>
</table>

## Funding
This work has been supported by The Henry L. Guenther Foundation, Sonderforschungsbereich (SFB) 974 (TP B07) of the German Research foundation, and the National Institute of Health, grants T32 AG00096, R00 AG062230, R21 EB033516, R01 EB016089, R01 EB023963, K00AG068440, P30 AG066519, R21 AG053040, R01 AG076942, P30 AG066519, and P41 EB031771.