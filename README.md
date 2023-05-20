# Multivariate Regression Method for Air Pollution Prediciton in a Given Area

![flowchart]("scripts/Flowchart1.png")

## Abstract
The increasing air pollution is causing significant risks to public health and the global economy. It caused premature deaths to seven million people (about fourteen times the population of Gdansk) in 2019. The issue emphasizes the need for reliable forecasting models to be developed. This thesis investigates a novel algorithm, combining multivariate regression, the sliding window approach, and the Moore-Penrose pseudoinverse to predict air pollution levels in Warsaw – a city with significant pollution. The research identifies a promising algorithm utilizing a 30-day sliding window for forecasting air pollution. This approach delivers encouraging results when integrated with the Least Absolute Shrinkage and Selection Operator (LASSO) method for predictor selection. The primary application of this model involves its integration into software applications, providing an accessible tool for individuals to plan outdoor activities, thereby mitigating health risks associated with high pollution levels. This research contributes significantly to environmental data science and enhances public access to vital air quality information by offering an innovative solution to a crucial global concern.


## Directories:
- /scripts
- /results of 29 studies
- /scripts/other graphs
- /scripts/Plots_exploring_the_dataset


## Files:
- thesis.pdf,
- scripts/apd21_air_pollution.py,
- scripts/apd51_weatehr_data.,
- scripts/exploration10.m,
- scripts/trail10.m,
- scripts/trail10summer.m,
- scripts/trail10winter.m.

### **thesis.pdf**
This document contain the text of my thesis.

### **apd21_air_pollution.py**
This program acquire air pollution data.

### **apd51_air_pollution.py**
This program acquire weather data.

### **warsaw_data.csv**
This is dataset file.

### **exploration10.m**
This script contain different methods for exploring the dataset.

### **trial10.m**
This script contain the method. It predicts air pollution based on the whole year period.

### **trial10summer.m**
This script contain the method. It predicts air pollution based on the summer period.

### **trial10winter.m**
This script contain the method. It predicts air pollution based on winter period.

The other files present in `scripts` directory are either functions required to run the files described above or consists of analysis support.

`results of 29 studies` contains the outcomes of conducting all of 29 studies on the dataset.

Directory `other graphs` contain various graphical aids used in the paper.

`Plots_exploring_the_dataset` is the folder which is created while executing of **exploration10.m** and contains plots supporting descriptive statistics.


<sub>Szymon Bartoszewicz WSB Merito University in Gdansk, 2023</sub>
