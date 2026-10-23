# Tissue parameters for photoacoustic simulation: sources

Compiled and fetched 2026-10-20; housekeeping 2026-10-21 (section 12; attributions in sections 3, 5, 6 and 8, and the CHECK and UNVERIFIED sections, brought into line with `tissue_params.json`; no value changed). Machine-readable values are in `tissue_params.json` in this directory. Every number below was read from a source retrieved in this session, or was computed here from such numbers with the arithmetic shown. Status labels: VERIFIED (read from a fetched source), DERIVED (computed here from fetched values), UNVERIFIED (no source retrieved, value null).

Conventions. Absorption and scattering coefficients are Napierian (natural log); 1 cm^-1 = 100 m^-1. Molar extinction coefficients are decadic, in cm^-1 M^-1; 1 cm^-1 M^-1 = 0.1 m^2 mol^-1. For attenuation, 1 Np = 8.686 dB.

Access limits that shaped this file. Direct downloads from omlc.org and itis.swiss were refused by the session proxy, so OMLC files were read through a page-fetch tool that quotes rows. The IT'IS Foundation tissue database pages render their tables in JavaScript and returned no numbers. Its V4.1 zip (DOI 10.13099/VIP21000-04-1) could not be downloaded. IT'IS values therefore appeared at first only second-hand, through a paper that cites the database. Duck 1990 (a book) and Beard 2011 (Interface Focus) were not retrieved in the first round. Later rounds changed this (section 12): Beard 2011 and Mast 2000 were fetched in gap-fill round 1, and the IT'IS V5.0 online tables were read in a browser.

## 1. Haemoglobin

Source: S. Prahl (1999), "Tabulated Molar Extinction Coefficient for Hemoglobin in Water", OMLC, https://omlc.org/spectra/hemoglobin/summary.html, compiled from data by W. B. Gratzer (MRC Labs, London) and N. Kollias (Wellman Laboratories, Harvard Medical School). The file's column header reads "Hb02" (with a zero), which is taken to mean HbO2.

| lambda (nm) | epsilon HbO2 (cm^-1 M^-1) | epsilon Hb (cm^-1 M^-1) | Location |
|---|---|---|---|
| 532 | 43876 | 40584 | summary.html row 532 |
| 750 | 518 | 1405.24 | row 750 |
| 800 | 816 | 761.72 | row 800 |
| 850 | 1058 | 691.32 | row 850 |

Conversion to whole blood. The Prahl index page (https://omlc.org/spectra/hemoglobin/) gives the formula mu_a(lambda) = (2.303) e(lambda) (150 g/liter)/(64,500 g Hb/mole) = 0.0054 e(lambda), with mu_a in cm^-1. It states "Assume 150 g Hb/liter" and uses "64,500 as the gram molecular weight of hemoglobin". Jacques 2013 (Phys. Med. Biol. 58 R37, doi 10.1088/0031-9155/58/11/R37), Sec. 5, Eq. (6), writes mu_a = ln(10) sum(C_i epsilon_i). It also assigns whole blood C_m,HGb = 150 g L^-1 but uses MW = 64 458 g mol^-1. The factor 2.303 is ln(10) = 2.302585 rounded.

Factor used here: ln(10) x 150 / 64500 = 0.00535485 cm^-1 per (cm^-1 M^-1). Prahl's 2.303 gives 0.00535581. Jacques's MW gives 0.00535834. The largest difference is 0.07%.

Erratum in the source. Jacques 2013 prints the factor as "0.0536 epsilon_blood". That is a factor of ten too large. Its own inputs and its worked result (4.38 cm^-1 for whole blood at 806 nm, epsilon = 818) require 0.00536.

| lambda (nm) | mu_a whole blood, 100% HbO2 (cm^-1) | mu_a whole blood, 100% Hb (cm^-1) |
|---|---|---|
| 532 | 234.95 | 217.32 |
| 750 | 2.774 | 7.525 |
| 800 | 4.370 | 4.079 |
| 850 | 5.665 | 3.702 |

These values are DERIVED. For saturation S, mu_a = S mu_a,HbO2 + (1 - S) mu_a,Hb (Jacques 2013 Eq. 12, with blood volume fraction B multiplying both terms for tissue).

## 2. Melanin

Two published power laws were found. They differ in exponent.

1. Jacques 2013, Sec. 5.3, Eq. (8), p. R53: mu_a.melanosome = 519 cm^-1 (lambda/500 nm)^-m. The 519 cm^-1 at 500 nm "was specified by the laser experiments", and "an approximate value for m is 3". It applies to the interior of a typical cutaneous melanosome. Tissue absorption is f_v,melanosome x mu_a.melanosome (Eq. 9). The same section gives eumelanin and pheomelanin molar forms: epsilon_eu = 2.37e4 exp(-0.0056 lambda), epsilon_pheo = 1.01e5 exp(-0.0087 lambda) cm^-1 M^-1 (Eqs. 10, 11).
2. S. L. Jacques (1998), "Melanosome Absorption Coefficient", OMLC, https://omlc.org/spectra/melanin/mua.html, section "A key lesson": mu_a = 1.70e12 x lambda[nm]^-3.48 cm^-1 for skin, and 6.49e12 x lambda^-3.48 cm^-1 for retina. The page notes a roughly ten-fold variation in melanin concentration within melanosomes. Its underlying data are Jacques and McAuliffe 1991 (Photochem. Photobiol. 53:769-775) and Jacques, Glickman and Schwartz 1996 (SPIE 2681:468-477).

| lambda (nm) | Jacques 2013 Eq. 8, m = 3 (cm^-1) | OMLC skin, ^-3.48 (cm^-1) | OMLC retina (cm^-1) |
|---|---|---|---|
| 532 | 430.87 | 554.98 | 2118.7 |
| 750 | 153.78 | 167.97 | 641.3 |
| 800 | 126.71 | 134.18 | 512.3 |
| 850 | 105.64 | 108.66 | 414.8 |

All of these are DERIVED and describe the interior of the melanosome, not bulk tissue. The two skin formulas agree within 9% from 750 to 850 nm and diverge by 29% at 532 nm.

## 3. Lipid and water

Fat. Source: R. L. P. van Veen, H. J. C. M. Sterenborg, A. Pifferi, A. Torricelli, R. Cubeddu, "Determination of VIS-NIR absorption coefficients of mammalian fat, with time- and spatially resolved diffuse reflectance and transmission spectroscopy". The OMLC file header says "OSA Annual BIOMED Topical Meeting, 2004". The PDF itself prints "(c)2000 Optical Society of America". The year is taken as 2004 (gap-fill round 1, `gapfill_round1.md` section 8): the PDF cites a 2002 paper in its reference [1], so it cannot date from 2000, and the 2000 line is a template copyright. Jacques 2013 also cites it as 2004. The archival journal version is van Veen et al. 2005, J. Biomed. Opt. 10(5), 054004. Data file: https://omlc.org/spectra/fat/fat.txt, processed by S. Prahl to integer wavelengths and three significant digits.

Units. The data file states no unit. The proceedings PDF (https://omlc.org/spectra/fat/vanveen04.pdf) labels its axes "absorption coefficient µa [m-1]". Values are therefore taken as m^-1. The sample is purified pig-lard oil. The OMLC page states that the error bars come from the standard deviation over 37 and 60 C. The paper's reported range is 600 to 1100 nm, so the 532 nm value lies outside it and carries lower confidence.

| lambda (nm) | mu_a fat (m^-1) | mu_a fat (cm^-1) | Row (fat.txt) | fat.orig.txt nearest |
|---|---|---|---|---|
| 532 | 1.002 | 0.01002 | 532 | 532.023: 1.001871214 |
| 750 | 0.972 | 0.00972 | 750 | 750.325: 0.982525117 |
| 800 | 0.403 | 0.00403 | 800 | 800.047: 0.40383996 |
| 850 | 0.637 | 0.00637 | 850 | 849.841: 0.63831235 |

File maximum: 13.10 m^-1 at 930 nm. Minimum: 0.310 m^-1 at 708 nm.

Water. Sources: G. M. Hale and M. R. Querry (1973), "Optical constants of water in the 200nm to 200 micron wavelength region", Appl. Opt. 12, 555-563, tabulated at https://omlc.org/spectra/water/data/hale73.txt. Also D. J. Segelstein (1981), "The complex refractive index of water", tabulated at https://omlc.org/spectra/water/data/segelstein81.txt. Both are in cm^-1.

| lambda (nm) | Hale and Querry (cm^-1) | Segelstein (cm^-1) | Notes |
|---|---|---|---|
| 532 | 0.0003564 (DERIVED) | 0.000427516 (DERIVED) | Hale rows 525: 0.00032, 550: 0.00045. Segelstein rows 530: 0.00041682, 535: 0.00044356. Linear interpolation. |
| 750 | 0.026000 | 0.026125 | exact rows |
| 800 | 0.020000 | 0.019640 | exact rows |
| 850 | 0.043000 | 0.0438418 (DERIVED) | Segelstein rows 849: 0.043343, 855: 0.046336 |

At 532 nm the two tabulations differ by 20%. Water absorption there is negligible against blood, so the difference does not matter for blood-bearing tissue. Jacques 2013 Sec. 5 gives one point check: mu_a.water = 0.45 cm^-1 at 970 nm.

## 4. Reduced scattering

Source: Jacques 2013, Eq. (1), Sec. 2: mu_s' = a (lambda/500 nm)^-b, where a is mu_s' at 500 nm and b is the scattering power. Table 2 is titled "Average parameters for reduced scattering coefficient, mu_s, for tissues" (p. R42). Its data file is https://omlc.org/news/dec14/Jacques_PMB2013/table2_JacquesPMB2013.csv.

| Tissue (Table 2) | a mean (cm^-1) | a SD | b mean | b SD | n |
|---|---|---|---|---|---|
| Skin | 46.0 | 13.7 | 1.421 | 0.517 | 8 |
| Brain | 24.2 | 11.7 | 1.611 | 1.063 | 8 |
| Breast | 16.8 | 8.1 | 1.055 | 0.771 | 8 |
| Bone | 22.9 | 14.6 | 0.716 | 0.682 | 3 |
| Other soft tissues | 18.9 | 10.2 | 1.286 | 0.521 | 18 |
| Other fibrous tissues | 27.1 | 5.0 | 1.627 | 0.115 | 5 |
| Fatty tissue | 18.4 | 9.0 | 0.672 | 0.242 | 6 |

Generic soft tissue is taken as a = 18.9 cm^-1 (1890 m^-1) and b = 1.286. Evaluated values (DERIVED): mu_s'(532) = 17.451, mu_s'(750) = 11.220, mu_s'(800) = 10.327 and mu_s'(850) = 9.552 cm^-1.

Anisotropy g. Jacques 2013 (Sec. 3, pp. R44 to R47) gives no single tissue value. It reports "values of g greater than 0.90 for visible wavelengths". For cell suspensions it cites g = 0.925, 0.950 and 0.930 (Mourant et al 2002), and Fig. 6 overlays a Henyey-Greenstein curve for g = 0.930. It also states that mu_s(1 - g) "remains somewhat constant in the range of 30-50 cm^-1" at 488 nm (Fig. 5), and that the mean free path 1/mu_s is "typically 100 um or less" (p. R44). Recorded range: 0.90 to 0.95. A single typical value is UNVERIFIED.

## 5. Transport mean free path (DERIVED)

At 800 nm, for generic soft tissue:

- mu_s' = 18.9 x (800/500)^-1.286 = 18.9 x 1.6^-1.286 = 10.327 cm^-1.
- mu_a = 0.0438 cm^-1, the Jacques 2013 Sec. 5 worked example for a 1% blood volume fraction at 806 nm.
- l_tr = 1/(mu_a + mu_s') = 1/(0.0438 + 10.327) cm = 1/10.371 cm = 0.0964 cm = 0.964 mm.

Sensitivity, with each figure computed here:

- With mu_a = 0, l_tr = 0.968 mm.
- Adding pure-water absorption at 800 nm (0.0196 cm^-1) as a full-water upper bound gives 0.962 mm.
- Over a mean plus or minus 1 SD (8.7 to 29.1 cm^-1), l_tr runs from 2.08 mm down to 0.63 mm.
- Over b plus or minus 1 SD (0.765 to 1.807), l_tr runs from 0.76 to 1.23 mm.
- Across the seven tissue-type means, l_tr runs from 0.42 mm (skin) to 0.97 mm (breast).
- For soft tissue at 750 nm l_tr = 0.891 mm, and at 850 nm 1.047 mm (mu_a neglected, which changes the result by less than 0.5%).

Literature statements, read as approximate (Zhou 2016 and Wang 2008 re-checked 2026-10-21). The number comes from Zhou, Yao and Wang 2016 (J. Biomed. Opt. 21, 061007), Sec. 4.2, p. 061007-5: the penetration of OR-PAM "is limited to about one transport mean free path in tissue (~1 mm)". Wang 2008 (IEEE J. Sel. Top. Quantum Electron. 14, 171-179, doi 10.1109/JSTQE.2007.913398), Sec. VIII, p. 177, gives only the correlation: "The 1 mm optical penetration limit is approximately correlated with the optical transport mean free path." It states no value for the transport mean free path. Wang and Hu 2012 (Science 335, 1458-1462, doi 10.1126/science.1216210), p. 1458: "up to ~1 mm, as defined by the optical diffusion limit".

## 6. Acoustic properties

Primary table found: M. O. Culjat, D. Goldenberg, P. Tewari, R. S. Singh (2010), "A review of tissue substitutes for ultrasound imaging", Ultrasound Med. Biol. 36(6), 861-873, doi 10.1016/j.ultrasmedbio.2010.02.012, Table 1. That table cites Mast 2000 and ICRU 1998. Neither was fetched in the first round; Mast 2000 was fetched in gap-fill round 1 and confirms every row used (section 12). ICRU 61 remains unfetched. Temperature is not stated in the fetched table.

| Material | c (m/s) | rho (kg/m^3) | alpha0 (dB cm^-1 MHz^-1) | Z (MRayl) | Table 1 source |
|---|---|---|---|---|---|
| Soft tissue (average) | 1561 | 1043 | 0.54 | 1.63 | Mast 2000 |
| Fat | 1478 | 950 | 0.48 | 1.40 | Mast 2000 |
| Muscle | 1547 | 1050 | 1.09 | 1.62 | Mast 2000 |
| Blood | 1584 | 1060 | 0.2 | 1.68 | ICRU 1998 |
| Liver | 1595 | 1060 | 0.5 | 1.69 | ICRU 1998 |
| Water | 1480 | 1000 | 0.0022 | 1.48 | none given |

Corroborating table: Stanford RAD225/BIOE225 course slides, "Acoustic Properties of Tissue" (Fall 2019), https://web.stanford.edu/class/rad225/subdir/Class3.pdf. Slide 19, credited to Szabo, gives (density, speed, impedance):

- Water at 22 C: 998, 1482, 1.48.
- Blood: 1060, 1584, 1.68.
- Skeletal muscle: 1041, 1580, 1.65.
- Liver: 1050, 1578, 1.64.
- Fat: 928, 1430, 1.33.

Slide 29, credited to Duck (not Szabo), gives attenuation (dB/cm/MHz) for water 0.0022, blood 0.14, liver 0.45, muscle 1 and soft tissue 0.5 (re-checked 2026-10-21). The 0.5 used as `alpha_soft.value_alt.Stanford_slides_soft_tissue` is therefore a Duck figure as quoted on the slides; Duck 1990 itself was not read. Wang 2008, Sec. V, p. 174 uses "sound velocity in soft tissue (1.54 mm/us)", and p. 172 gives "~1480 m/s in water".

Values adopted for generic soft tissue:

- Speed of sound: 1561 m/s (range 1430 to 1595 m/s).
- Density: 1043 kg/m^3 (range 928 to 1060).
- Impedance: 1.63 MRayl (check: 1561 x 1043 = 1.628e6 kg m^-2 s^-1).
- Attenuation slope: 0.54 dB cm^-1 MHz^-1 (= 6.217 Np m^-1 MHz^-1), with a tissue range of 0.2 (blood) to 1.09 (muscle).

Frequency dependence, alpha(f) = alpha0 f^y:

- The k-Wave documentation (https://www.k-wave.org/documentation/example_na_modelling_absorption.php) states that soft-tissue absorption follows a power law "in which the exponent is between 1 and 2".
- Parker, "Power laws prevail in medical ultrasound" (arXiv:2203.04239), Sec. 2.4: "a power law fit near 1 was adequate for attenuation models of soft tissues". Its Table 1 gives alpha0 = 0.05 Np/cm-MHz (0.434 dB/cm/MHz) with exponent about 1.
- Kelly, McGough and Meerschaert 2008 (J. Acoust. Soc. Am. 124, 2861-2872), Sec. IV, quote two media from Duck. A liver-like medium has y = 1.139 and alpha0 = 0.0459 Np cm^-1 MHz^-1.139. A fat-like medium has y = 1.5 and alpha0 = 0.086 Np cm^-1 MHz^-1.5.
- Suomi et al. 2018 (arXiv:1811.02874) use y = 1.1 in their Table 1 header.

Water absorption scales with f^2, not f. From the Pinkerton 1949 polynomial in k-Wave `waterAbsorption.m`, it is 0.00206 dB/cm at 1 MHz and 22 C, and 0.00139 dB/cm at 1 MHz and 37 C (DERIVED).

## 7. Thermal properties

Water at 37 C (310.15 K, 0.101325 MPa). Source: NIST Chemistry WebBook SRD 69, "Isobaric Properties for Water". It implements Wagner and Pruss, IAPWS-95, J. Phys. Chem. Ref. Data 31, 387-535 (2002). Row T = 310.150 K: rho = 993.330 kg/m^3, c_p = 4.17924 J/(g K), c_v = 4.08695 J/(g K), sound speed 1523.66 m/s and k = 0.624475 W/(m K). Thermal diffusivity (DERIVED) is 0.624475 / (993.330 x 4179.24) = 1.504e-7 m^2/s.

Tissue values:

- Wang 2008, p. 172: thermal diffusivity "~0.1 mm2/s for tissue" (1.0e-7 m^2/s). Specific heat "[~4000 J/(kg K) for muscle]". Isothermal compressibility "~5 x 10^-10 Pa^-1 for water or soft tissue". Thermal expansion "~4 x 10^-4 K^-1 for muscle".
- Suomi et al. 2018, Tables 1 and 2, citing Hasgall et al. 2015 (IT'IS database v3.0) and Mast 2000. Muscle: k = 0.49 W/(m K), c = 3421 J/(kg K), rho = 1050 kg/m^3. Fat: k = 0.21, c = 2348, rho = 950. Blood: c = 3617. Water: k = 0.60, c = 4178. Temperature is not stated. Derived diffusivities are 1.364e-7 m^2/s for muscle and 9.41e-8 m^2/s for fat. These values are second-hand IT'IS figures.
- Bianchi et al. 2022 (Int. J. Hyperthermia 39, 297-340, doi 10.1080/02656736.2022.2028908), Sec. 5.1: "most of thermal conductivity values of biological tissues and media range between 0.4 W/(mK) and 0.7 W/(mK)" (20 to 60 C). For porcine liver near 35 to 40 C it gives diffusivity of about 0.141e-6 m^2/s (Guntur et al., as cited). The fetched text was partly garbled, so these figures need confirming against the PDF.

## 8. Grueneisen parameter

Definition: Gamma = beta c^2 / c_p = beta / (kappa rho c_v). See Wang 2008 Eq. (7) and Yao et al. 2014, Introduction.

Water, computed here from IAPWS-95 (NIST WebBook), with beta taken from a central difference of density over plus or minus 1 K (DERIVED):

| T (C) | beta (K^-1) | c (m/s) | c_p (J/(kg K)) | Gamma |
|---|---|---|---|---|
| 20 | 2.069e-4 | 1482.35 | 4184.05 | 0.1086 |
| 22 | 2.275e-4 | 1488.33 | 4182.78 | 0.1205 |
| 37 | 3.619e-4 | 1523.66 | 4179.24 | 0.2010 |

Published water values:

- Bakaric et al. 2019 (IEEE IUS, Glasgow; https://discovery-pp.ucl.ac.uk/10087155/1/2019-Bakaric-IEEEIUS.pdf) give Gamma(T) = 0.0053 T + 0.0043. They describe it as an approximation "derived from the definition of the Gruneisen parameter" (their Eq. 2) together with the temperature dependence of sound speed, thermal expansion and specific heat "for water and aqueous solutions [14]". Reference [14] is Wang and Wu, Biomedical Optics: Principles and Imaging, Wiley 2012. The formula is therefore Bakaric et al.'s derivation from property data cited to Wang and Wu, not a formula given by Wang and Wu (re-checked 2026-10-21). It gives 0.1103 at 20 C, 0.1209 at 22 C and 0.2004 at 37 C. The source compares it with measurements over 22 to 50 C and states no range for the formula itself.
- Petrova et al. 2013 (Opt. Express 21, 25077, doi 10.1364/OE.21.025077): 0.11 "at room temperature".
- Yao et al. 2014 (J. Biomed. Opt. 19, 017007, doi 10.1117/1.JBO.19.1.017007), Sec. 2.2: 0.12 at 22 C.
- Yao and Wang 2013 (Proc. SPIE 8581, 858138): 0.13 at 24 C.

The independent IAPWS-95 computation agrees with the linear formula to within 0.002 at 20, 22 and 37 C.

Measured tissue values from Yao et al. 2014, all at 22 C:

- Porcine subcutaneous fat: 0.81 plus or minus 0.05 (n = 6).
- Porcine lipid (lard): 0.69 plus or minus 0.02 (n = 5).
- Bovine serum: 0.132 plus or minus 0.002.
- Red blood cell suspensions: 0.129 (15.9 mg/ml Hb) to 0.144 (59.3 mg/ml), with the fit Gamma = 0.124 + 0.000333 C_HbO2 [mg/ml].
- Whole blood at 110 mg/ml: estimated at 0.16.

Soft tissue. Wang 2008's approximate inputs (beta 4e-4 K^-1, kappa 5e-10 Pa^-1, rho 1000 kg/m^3, c_v 4000 J/(kg K)) give Gamma = 4e-4 / (5e-10 x 1000 x 4000) = 0.20 (DERIVED, order of magnitude only). For temperature dependence, Mahmoodkalayeh et al. (arXiv:1802.07114) cite Larina et al. 2005 (J. Phys. D 38, 2633) for a "1.5% C-1 increase in the Grueneisen parameter in a biological tissue". Wang and Hu 2012, p. 1462, report a photoacoustic pressure increase of "~5% per degree" in tissue phantoms. A measured Gamma for non-fatty solid soft tissue at 37 C is UNVERIFIED: no source read reports one. Whole blood at 37 C was later measured by Villanueva-Palero et al. 2017 (section 12).

## 9. Cell diameter

- BioNumbers BNID 100432 (https://bionumbers.hms.harvard.edu/bionumber.aspx?id=100432): "Cell diameter", mammalian tissue culture cell, 15 um, range 10 to 20 um. Primary source: "Roskams and Rodgers, LabRef".
- BNIDs 108870 and 108871: HeLa cells in suspension (Coulter Multisizer) measure about 17.5 um at 3 days and about 15.0 um at 5 days after seeding. Source: Luciani et al. 2001, Eur. J. Cell Biol. 80:187-95, Fig. 2a.
- BNID 108239: human red blood cell, 7.7 um.
- BNID 109546: human lymphocyte, about 7 um (Marguerat and Bahler 2012, Trends Genet. 28:560-5).
- BNID 104990: HeLa nucleus 0.22 pl plus cytoplasm 0.94 pl. That total gives an equivalent-sphere diameter of 13.0 um (DERIVED, d = (6V/pi)^(1/3)).

## 10. Refractive index and speed of light

- c0 = 299 792 458 m/s, exact. Source: NIST CODATA 2022, https://physics.nist.gov/cuu/Constants/Table/allascii.txt.
- Jacques 2013, Sec. 4, Eq. (3), p. R47: n = n_dry - (n_dry - n_water) W, with n_water = 1.33 and n_dry = 1.50 (Jacques and Prahl 1987). An alternative is n_dry = 1.514 (Biswas and Luu 2011). W is the water volume fraction. No soft-tissue W was sourced. Illustrative values: W = 0.6 gives 1.398, W = 0.7 gives 1.381 and W = 0.8 gives 1.364.
- Giannios et al. 2016, "Visible to near-infrared refractive properties of freshly-excised human-liver tissues" (Sci. Rep. 6, 27910, doi 10.1038/srep27910), Table 2, normal liver at 24 plus or minus 2 C: 1.378 plus or minus 0.008 at 632.8 nm, 1.369 plus or minus 0.005 at 964 nm and 1.362 plus or minus 0.003 at 1551 nm. Cauchy coefficients: A = 1.35910, B = 0.00827 um^2, C = -0.000576 um^4. Assuming the form n = A + B/lambda^2 + C/lambda^4 (the equation is an image in the source, so the form is inferred from the coefficient units) gives n(800 nm) = 1.3706 (DERIVED). Re-checked 2026-10-21 by reading PMC4906272 in a browser, Table 2, normal tissue group N: every value matches, and 1.35910 + 0.00827/0.64 - 0.000576/0.4096 = 1.3706.
- Speed of light in tissue (DERIVED): c = c0 / n = 299792458 / 1.37 = 2.1883e8 m/s. Over n = 1.364 to 1.398 it runs from 2.1979e8 to 2.1444e8 m/s. In water (n = 1.33) it is 2.2541e8 m/s.

## 11. Layered media: water at 37 C, fat, muscle

| Medium | c (m/s) | rho (kg/m^3) | Z = rho c (MRayl) | Source and location |
|---|---|---|---|---|
| Water, 37 C | 1523.66 | 993.330 | 1.5135 (DERIVED) | NIST WebBook IAPWS-95, row 310.150 K |
| Water, 37 C (check) | 1523.93 | 993.325 | 1.5137 | Marczak 1997 polynomial (J. Acoust. Soc. Am. 102, 2776) and Jones and Harris 1992 (J. Res. NIST 97, 335), as coded in k-Wave `waterSoundSpeed.m` and `waterDensity.m` (GitHub ucl-bug/k-wave), evaluated at 37 C |
| Fat | 1478 | 950 | 1.40 (check 1.4041) | Culjat 2010 Table 1 (Mast 2000) |
| Fat (alternative) | 1430 | 928 | 1.33 | Stanford slide 19 (Szabo) |
| Muscle | 1547 | 1050 | 1.62 (check 1.6244) | Culjat 2010 Table 1 (Mast 2000) |
| Skeletal muscle (alternative) | 1580 | 1041 | 1.65 | Stanford slide 19 (Szabo) |

The Marczak coefficients as coded in k-Wave, highest power first, are 2.787860e-9, -1.398845e-6, 3.287156e-4, -5.779136e-2, 5.038813 and 1.402385e3. The two water routes agree within 0.27 m/s at 37 C.

## 12. Later rounds (added 2026-10-21)

### Gap-filling round 1 (2026-10-20)

Records and text are in `gapfill_round1.json` (39 records) and `gapfill_round1.md`. That round did not edit this file or `tissue_params.json`; the points below summarise it. Every shell download from scholarly hosts was refused, so all retrieval went through the page-fetch tool.

- Grueneisen parameter near 37 C (gap 1). No measured value for non-fatty soft tissue was found. Every absolute tissue value is at 22 to 24 C, and every tissue temperature study is relative. Liang et al. 2018 measure pure substances at 25, 35 and 45 C; distilled water interpolates to 0.1932 at 37 C. A DERIVED bounding estimate for non-fatty soft tissue at 37 C is 0.16 to 0.25. Six scalings of the Yao 2014 bovine-serum value (0.132 at 22 C) give 0.162 to 0.253, and the definition Gamma = beta c^2 / c_p with second-hand inputs gives 0.182 to 0.268. The 0.22 suggested there is an analytical choice, not a sourced figure. DERIVED extrapolations: blood at 37 C 0.16 to 0.18, fat 0.61 to 0.74.
- Beard 2011 (gap 2). Read through the PMC path `/articles/3262268`. It gives a temperature rise below about 0.1 K, pressures below 10 kPa, a skin MPE of 20 mJ cm^-2 at 670 nm, mu_eff about 0.13 mm^-1, penetration about 8 mm and 0.75 dB cm^-1 MHz^-1. It gives no Grueneisen value and no numerical confinement criterion.
- Duck 1990, second-hand (gap 3). Six Duck-derived tissues at 37 C via Mast 2000; liver, muscle, brain and fat via Chatelin et al.; tissue thermal expansion via Shah 2008. No Duck table number was found.
- IT'IS (gap 4). At that round, values were available only second-hand: fat, muscle and skin rho, c_p and k (v3.0 via the COMSOL Bioheat library). The direct read came later (below).
- Mast 2000 (gap 5). Fetched from the author's copy at homepages.uc.edu. Its Table 1 covers 22 tissues. The mean of 1561 m/s, 1043 kg/m^3 and 0.54 dB/cm at 1 MHz confirms the Culjat soft-tissue average and every Culjat row. ICRU 61 itself was not retrieved.
- Anisotropy g (gap 6). The median of eight measured soft-tissue values at 790 to 850 nm (Cheong, Prahl and Welch tables) is 0.9485, range 0.926 to 0.968 (DERIVED).
- Water fraction (gap 7). Mass fractions: muscle 0.76, skin 0.72, blood 0.80 (JASN 2011 Table 3). Volume fraction: dermis 0.65 in vivo (Jacques 2013 Table 3).
- van Veen year and units (gap 8). Year 2004; unit m^-1 confirmed by the JBO 2005 journal version. The fat records at 532, 750, 800 and 850 nm now carry year 2004 with the same year note as `mua_fat_700`.
- Water at 532 nm (gap 9). Pope and Fry 1997 give 0.0004444 cm^-1 (DERIVED by interpolation), which supports Segelstein over Hale and Querry.
- Culjat temperature (gap 10). Not stated, and Mast 2000 states it only for the Duck-derived rows.

### Browser round (2026-10-20): two records added

- `gamma_blood_human_37C_measured`. Villanueva-Palero, Hondebrink, Petersen and Steenbergen 2017, Photoacoustics 6:9-15, doi 10.1016/j.pacs.2017.03.004, PMC5388938, read in the built-in browser after automated fetching was blocked. The measurement is fresh human whole blood at 750 nm. At 37 C (Table 6, five donors) the mean is 0.226 with a between-donor SD of 0.015; at 22 C (Table 4) the mean is 0.162 with an SD of 0.016. The authors did not correct the acoustic impedance mismatch between calibration ink and blood. This is whole blood, not solid soft tissue.
- `itis_v5_direct`. IT'IS Foundation Tissue Properties Database V5.0, DOI 10.13099/VIP21000-05-0, online per-property tables read directly in the built-in browser. Examples: muscle sound speed 1588.4 m/s and fat 1440.2 m/s; fat density 911 kg/m^3 and muscle 1090; muscle k 0.49 W/(m K) and C_p 3421 J/(kg K); fat k 0.21 and C_p 2348. The muscle and fat k and C_p match the second-hand values in section 7. The attenuation table gives no reference frequency in the extracted text.

The same tissue-solver work on 2026-10-20 added the 700 nm records (`eps_HbO2_700`, `eps_Hb_700`, `mua_blood_HbO2_700`, `mua_blood_Hb_700`, `mua_water_700`, `mua_fat_700`, `mua_mel_700`, `musp_soft_700`) for the 700 nm out-of-distribution wavelength. They are recorded in the JSON change log and are not repeated in sections 1 to 4.

### Housekeeping (2026-10-21)

Text-only corrections to the JSON (audit round 1, MINOR 13, and the drafting review). No value changed; `check_values_unchanged.py` asserts this against `tissue_params.before_housekeeping.json`. The Stanford slide-29 attenuation table is now credited to Duck. Zhou 2016 is cited for the ~1 mm transport mean free path and Wang 2008 only for the correlation. The Bakaric formula is described as Bakaric's own derivation. The fat year is 2004. `n_liver_NIR` carries its 2026-10-21 re-check. The unverified list and change log are brought up to date.

## CHECK

Three figures were stated earlier in this project without being looked up. Each is compared below with what was found.

(a) Speed of sound in soft tissue about 1,540 m/s. Verdict: CONFIRMED WITH QUALIFICATION. Wang 2008 (p. 174) uses 1540 m/s for soft tissue. The only sourced soft-tissue average found, Mast 2000 via Culjat 2010 Table 1, is 1561 m/s, 1.4% higher. Individual tissues span 1430 m/s (fat, Szabo via Stanford slide 19) to 1595 m/s (liver, ICRU 1998 via Mast 2000). The IT'IS V5.0 tables read later give muscle 1588 m/s and fat 1440 m/s. 1540 m/s is a conventional reference value inside that range, not a measured mean.

(b) Transport mean free path about 1 mm in NIR tissue. Verdict: CONFIRMED WITH QUALIFICATION. The sourced generic soft-tissue mean gives 0.964 mm at 800 nm (0.891 mm at 750 nm, 1.047 mm at 850 nm). Zhou 2016 states about 1 mm; Wang 2008 states only that the 1 mm penetration limit is approximately correlated with it. Between tissue types it falls to 0.42 mm in skin. Within plus or minus 1 SD of the soft-tissue scattering amplitude it spans 0.63 to 2.08 mm. Treat "1 mm" as an order-of-magnitude figure with a factor-of-two spread.

(c) A single cell is 10 to 20 micrometres across. Verdict: CONFIRMED WITH QUALIFICATION. BioNumbers BNID 100432 gives exactly 10 to 20 um (typical 15 um) for mammalian tissue-culture cells, and HeLa measures 15.0 to 17.5 um. Common human cells fall below this range: red blood cells are 7.7 um and lymphocytes about 7 um. The range describes typical cultured cells, not every mammalian cell.

## UNVERIFIED and gaps

Updated 2026-10-21 to match the `unverified` list in `tissue_params.json`. Items resolved since the first round are listed afterwards, with what resolved them.

Still open:

- IT'IS attenuation constant: its reference frequency, frequency exponent and measurement temperature. The V5.0 online tables were read directly (record `itis_v5_direct`), but the extracted attenuation table states no frequency. The v4.1 documentation gives no temperature for acoustic or thermal properties.
- Duck 1990, "Physical Properties of Tissue", read directly, with table numbers. Not retrieved. Duck-derived figures appear only second-hand: Mast 2000 Table 1, Chatelin et al., Shah 2008, Stanford slide 29 and Kelly et al. 2008.
- Beard 2011: the reference list (refs 78, 156 to 158 and 181), and whether the 1.6 mm^-1 in Sec. 3.1 is mu_s or mu_s'. The article text was read in gap-fill round 1; the PDF was not checked for the prime.
- ICRU Report 61 (1998) and Mast et al. 1997, read directly, and the temperature of the Culjat/Mast fat, muscle, blood and liver rows. Mast 2000 states no temperature for these rows.
- A measured Grueneisen parameter for non-fatty solid soft tissue at 37 C. The value is null. The gap-fill bounding range of 0.16 to 0.25 is DERIVED. The Villanueva-Palero blood values are for whole blood, not solid tissue.
- A single typical anisotropy g stated by a source. Jacques 2013 gives only "greater than 0.90" and 0.925 to 0.950 for cell suspensions. The gap-fill median of 0.95 is DERIVED from eight Cheong-table values.
- A water volume fraction W for generic soft tissue, as input to Jacques Eq. (3). No generic value exists in the sources read; per-tissue values were found in gap-fill round 1. The W values shown are illustrative.
- The temperature of the Stanford acoustic tables. It is not stated in the fetched text.
- Larina et al. 2005, the primary source of the 1.5% per C figure. Only the abstract was read, and it gives no coefficient.

Resolved since the first round, and no longer listed as unverified:

- IT'IS values read directly: the V5.0 online tables were read in the built-in browser on 2026-10-20 (record `itis_v5_direct`). Muscle and fat k and C_p match the second-hand values already used.
- Beard 2011 article text: read in gap-fill round 1 through the PMC path `/articles/3262268`. It gives no numerical Grueneisen value.
- Mast 2000 (ARLO 1:37): fetched in gap-fill round 1 from the author's copy. Its Table 1 confirms every Culjat row used.
- Fat absorption unit and year: the unit m^-1 is confirmed by the JBO 2005 journal version, and the year is 2004 (gap-fill round 1, section 8).
- Giannios 2016 Table 2 (`n_liver_NIR`): re-read on 2026-10-21 in a browser (PMC4906272). All values match. The audit had listed it as unconfirmed because its own re-fetch failed.
