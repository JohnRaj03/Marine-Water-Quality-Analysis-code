# Marine Water Quality Analysis and Predicting Optimal Fishing Grounds Using Machine Learning and Remote Sensing

## Project Overview
This project focuses on the **Cornish Coast, UK**, utilizing 10 years of NASA MODIS satellite data to monitor marine health and predict **Potential Fishing Zones (PFZs)**. By integrating remote sensing with machine learning, the study aims to support sustainable fisheries and marine habitat protection by identifying the environmental factors that influence fish distribution.

---

## Key Findings
* **Multi-Parameter Assessment:** Successfully tracked five critical oceanographic indicators: 
    * Chlorophyll-a
    * Sea Surface Temperature (SST)
    * Turbidity (Kd_490)
    * Particulate Organic Carbon (POC)
    * Particulate Inorganic Carbon (PIC)
* **Ideal Water Quality Ranges:** Identified specific parameter thresholds essential for maintaining a healthy marine ecosystem and supporting fish populations.
* **Predictive Accuracy:** The implementation of the **Random Forest** algorithm demonstrated high efficiency in finding patterns within complex remote sensing datasets to forecast PFZs.
* **Sustainable Impact:** The model provides actionable data to help fishermen reduce fuel usage and ecological footprints by accurately identifying productive shoaling areas.

---

## Study Methodology
1.  **Data Acquisition & Automation:** Automated retrieval of 10 years of MODIS Level-2 satellite imagery via Python’s `earthaccess` library and NASA’s OceanColor Web.
2.  **Image Processing & Validation:** Leveraged **NASA SeaDAS** for atmospheric correction, cloud masking, and rigorous data validation.
3.  **Integrated Analytics:** Combined statistical, geospatial, and time-series analysis to understand seasonal fluctuations and long-term environmental trends.
4.  **Predictive Modeling:** Developed a machine learning pipeline using the Random Forest method to correlate environmental variables with potential fish aggregation.

---

## Repository Contents
* **Python Scripts:** Automated data directly downloaded from 'earthaccess' server, cleaning, and multi-parameter visualization.
* **SeaDAS Project Files:** Processed satellite imagery and validation reports.
* **Predictive Model:** Trained Random Forest model for PFZ identification.
* **Final Dissertation:** Full documentation of research methodology and results.

---

## Technologies Used
* **Programming:** Python (Xarray, Pandas, NumPy, Matplotlib, earthaccess)
* **Software:** NASA SeaDAS (SeaWiFS Data Analysis System)
* **Machine Learning:** Random Forest Algorithm
* **Data Source:** NASA MODIS Satellite Sensors

---

## Next Steps
* **Sensor Expansion:** Incorporating data from additional satellite sensors to improve spatial resolution.
* **Time Integration:** Integrating real-time fish catch records to further refine model accuracy.
* **Species Identification:** Developing advanced algorithms to predict specific fish species distributions.

## Collaboration
I am open to discussions regarding marine data science, remote sensing, and sustainable resource management. If you are working on similar oceanographic projects or have insights into improving predictive models for marine conservation, **let's connect!**
