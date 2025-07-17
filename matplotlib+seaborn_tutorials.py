"""Matplotlib is a low level graph plotting library in python that 
serves as a visualization utilityatplotlib and Seaborn are essential for visualizing data before applying Machine Learning models.
📌 Matplotlib → For basic & customizable plots
📌 Seaborn → For statistical and beautiful visualizations

Both are must-haves for Data Analytics, Machine Learning, and Dashboards."""
#Checking Matplotlib Version
import matplotlib
print(matplotlib.__version__)

"""he plot() function is used to draw points (markers) in a diagram.
By default, the plot() function draws a line from point to point."""
#Draw a line in a diagram from position (1, 3) to position (8, 10):
import matplotlib.pyplot as plt
import numpy as np
xpoints = np.array([1, 8])
ypoints = np.array([3, 10])
plt.plot(xpoints, ypoints)
plt.show()

#Plotting Without Line
#Draw two points in the diagram, one at position (1, 3) and one in position (8, 10):
import matplotlib.pyplot as plt
import numpy as np
xpoints = np.array([1, 8])
ypoints = np.array([3, 10])
plt.plot(xpoints, ypoints, 'o')
plt.show()

#Multiple Points
#You can plot as many points as you like, just make sure you have the same number of points in both axis.
#Draw a line in a diagram from position (1, 3) to (2, 8) then to (6, 1) and finally to position (8, 10):
import matplotlib.pyplot as plt
import numpy as np
xpoints = np.array([1, 2, 6, 8])
ypoints = np.array([3, 8, 1, 10])
plt.plot(xpoints, ypoints)
plt.show()

#Default X-Points
"""If we do not specify the points on the x-axis, they will get the default values 0, 1, 2, 3 etc.,
depending on the length of the y-points."""
#Plotting without x-points:
import matplotlib.pyplot as plt
import numpy as np
ypoints = np.array([3, 8, 1, 10, 5, 7])
plt.plot(ypoints)
plt.show()

#Markers
"""You can use the keyword argument marker to emphasize each point with a specified marke"""
#Mark each point with a circle:
import matplotlib.pyplot as plt
import numpy as np
ypoints = np.array([3, 8, 1, 10])
plt.plot(ypoints, marker = 'o')#*
plt.show()

"""
Marker Reference
You can choose any of these markers:

Marker	Description
'o'	Circle	
'*'	Star	
'.'	Point	
','	Pixel	
'x'	X	
'X'	X (filled)	
'+'	Plus	
'P'	Plus (filled)	
's'	Square	
'D'	Diamond	
'd'	Diamond (thin)	
'p'	Pentagon	
'H'	Hexagon	
'h'	Hexagon	
'v'	Triangle Down	
'^'	Triangle Up	
'<'	Triangle Left	
'>'	Triangle Right	
'1'	Tri Down	
'2'	Tri Up	
'3'	Tri Left	
'4'	Tri Right	
'|'	Vline	
'_'	Hline
"""

"""
Format Strings fmt
You can also use the shortcut string notation parameter to specify the marker.
This parameter is also called fmt, and is written with this syntax:
marker|line|color
"""
#Mark each point with a circle:
import matplotlib.pyplot as plt
import numpy as np
ypoints = np.array([3, 8, 1, 10])
plt.plot(ypoints, 'o:r')
plt.show()
"""
Line Reference
Line Syntax	Description
'-'	Solid line	
':'	Dotted line	
'--'	Dashed line	
'-.'	Dashed/dotted line
"""

"""
Color Reference
Color Syntax	Description
'r'	Red	
'g'	Green	
'b'	Blue	
'c'	Cyan	
'm'	Magenta	
'y'	Yellow	
'k'	Black	
'w'	White
"""

"""Marker Size
You can use the keyword argument markersize or the shorter version, 
ms to set the size of the markers:"""
#Set the size of the markers to 20:
import matplotlib.pyplot as plt
import numpy as np
ypoints = np.array([3, 8, 1, 10])
plt.plot(ypoints, marker = 'o', ms = 20)
plt.show()

"""
Marker Color
You can use the keyword argument markeredgecolor or the shorter
mec to set the color of the edge of the markers:
"""
#Set the EDGE color to red:

import matplotlib.pyplot as plt
import numpy as np
ypoints = np.array([3, 8, 1, 10])
plt.plot(ypoints, marker = 'o', ms = 20, mec = 'r')
plt.show()

"""
You can use the keyword argument markerfacecolor or the 
shorter mfc to set the color inside the edge of the markers:
"""
#Set the FACE color to red:
import matplotlib.pyplot as plt
import numpy as np
ypoints = np.array([3, 8, 1, 10])
plt.plot(ypoints, marker = 'o', ms = 20, mfc = '#4CAF50', mec = 'hotpink')
plt.show()

#Matplotlib Line
"""
Linestyle
You can use the keyword argument linestyle, or shorter ls, to change the style of the plotted line:
"""
import matplotlib.pyplot as plt
import numpy as np
ypoints = np.array([3, 8, 1, 10])
plt.plot(ypoints, linestyle = 'dotted')
plt.show()

"""
Line Styles
You can choose any of these styles:

Style	Or
'solid' (default)	'-'	
'dotted'	':'	
'dashed'	'--'	
'dashdot'	'-.'	
'None'	'' or ' '
"""

"""
Line Color
You can use the keyword argument color or 
the shorter c to set the color of the line:
"""
import matplotlib.pyplot as plt
import numpy as np
ypoints = np.array([3, 8, 1, 10])
plt.plot(ypoints, color = 'r')
plt.show()

"""
Line Width
You can use the keyword argument linewidth or 
the shorter lw to change the width of the line.
The value is a floating number, in points:
"""
#Plot with a 20.5pt wide line:

import matplotlib.pyplot as plt
import numpy as np
ypoints = np.array([3, 8, 1, 10])
plt.plot(ypoints, linewidth = '20.5')
plt.show()

"""
Multiple Lines
You can plot as many lines as you like by simply adding more plt.plot() functions:
"""
#Draw two lines by specifying a plt.plot() function for each line:
import matplotlib.pyplot as plt
import numpy as np
y1 = np.array([3, 8, 1, 10])
y2 = np.array([6, 2, 7, 11])
plt.plot(y1)
plt.plot(y2)
plt.show()

"""Position the Title
You can use the loc parameter in title() to position the title."""
import numpy as np
import matplotlib.pyplot as plt

x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])
plt.title("Sports Watch Data", loc = 'left')
plt.xlabel("Average Pulse")
plt.ylabel("Calorie Burnage")
plt.plot(x, y)
plt.grid() #With Pyplot, you can use the grid() function to add grid lines to the plot.
plt.show()

#1.2 Basic Plotting
x = np.linspace(0, 10, 100)  # 100 points between 0 and 10
y = np.sin(x)
plt.plot(x, y, label="Sine Wave", color='b', linestyle='--', marker='o')
plt.xlabel("X Axis")
plt.ylabel("Y Axis")
plt.title("Line Plot Example")
plt.legend()
plt.grid(axis = 'x')
plt.grid(axis = 'y')
plt.show()

# Scatter Plot Example
x = np.random.rand(50)
y = np.random.rand(50)
plt.scatter(x, y, color='red', marker='*')
plt.title("Scatter Plot Example")
plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)
plt.show()

#Display Multiple Plots
#With the subplot() function you can draw multiple plots in one figure:
import matplotlib.pyplot as plt
import numpy as np
#plot 1:
x = np.array([0,1,2,3])
y = np.array([3,8,9,10])
plt.subplot(1,2,1)
plt.plot(x,y)
#plot 2:
x = np.array([0,2,1,4])
y = np.array([10,20,30,40])
plt.subplot(1,2,2)
plt.plot(x,y)
plt.show()

"""The subplot() Function
The subplot() function takes three arguments that describes the layout of the figure.
The layout is organized in rows and columns, which are represented by the first and second argument.
The third argument represents the index of the current plot.
plt.subplot(1, 2, 1)
#the figure has 1 row, 2 columns, and this plot is the first plot."""
"""So, if we want a figure with 2 rows an 1 column (meaning that the two plots will be 
displayed on top of each other instead of side-by-side), we can write the syntax like this:"""

import matplotlib.pyplot as plt
import numpy as np

x = np.array([0, 1, 2, 3])
y = np.array([3, 8, 1, 10])

plt.subplot(2, 3, 1)
plt.plot(x,y)

x = np.array([0, 1, 2, 3])
y = np.array([10, 20, 30, 40])

plt.subplot(2, 3, 2)
plt.plot(x,y)

x = np.array([0, 1, 2, 3])
y = np.array([3, 8, 1, 10])

plt.subplot(2, 3, 3)
plt.plot(x,y)

x = np.array([0, 1, 2, 3])
y = np.array([10, 20, 30, 40])

plt.subplot(2, 3, 4)
plt.plot(x,y)

x = np.array([0, 1, 2, 3])
y = np.array([3, 8, 1, 10])

plt.subplot(2, 3, 5)
plt.plot(x,y)

x = np.array([0, 1, 2, 3])
y = np.array([10, 20, 30, 40])

plt.subplot(2, 3, 6)
plt.plot(x,y)

plt.show()

"""reating Scatter Plots
With Pyplot, you can use the scatter() function to draw a scatter plot."""
import matplotlib.pyplot as plt
import numpy as np
x = np.random.rand(10)
y = np.random.rand(10)
plt.scatter(x,y)
plt.show()

#Draw two plots on the same figure:

import matplotlib.pyplot as plt
import numpy as np

#day one, the age and speed of 13 cars:
x = np.array([5,7,8,7,2,17,2,9,4,11,12,9,6])
y = np.array([99,86,87,88,111,86,103,87,94,78,77,85,86])
colors = np.array(["red","green","blue","yellow","pink","black","orange","purple","beige","brown","gray","cyan","magenta"])
plt.scatter(x, y, c=colors)

#day two, the age and speed of 15 cars:
x = np.array([2,2,8,1,15,8,12,9,7,3,11,4,7,14,12])
y = np.array([100,105,84,105,90,99,90,95,94,100,79,112,91,80,85])
plt.scatter(x, y,color = '#88c999')
plt.show()

"""How to Use the ColorMap
You can specify the colormap with the keyword argument cmap with the value of the colormap, 
in this case 'viridis' which is one of the built-in colormaps available in Matplotlib."""
import matplotlib.pyplot as plt
import numpy as np

x = np.array([5,7,8,7,2,17,2,9,4,11,12,9,6])
y = np.array([99,86,87,88,111,86,103,87,94,78,77,85,86])
colors = np.array([0, 10, 20, 30, 40, 45, 50, 55, 60, 70, 80, 90, 100])
plt.scatter(x, y, c=colors, cmap='viridis')
plt.colorbar() #colormap in the drawing by including the plt.colorbar() statement
plt.show()

#Set your own size for the markers:
#You can adjust the transparency of the dots with the alpha argument.
import matplotlib.pyplot as plt
import numpy as np

x = np.array([5,7,8,7,2,17,2,9,4,11,12,9,6])
y = np.array([99,86,87,88,111,86,103,87,94,78,77,85,86])
sizes = np.array([20,50,100,200,500,1000,60,90,10,300,600,800,75])

plt.scatter(x, y, s=sizes)

plt.show()

"""Combine Color Size and Alpha
You can combine a colormap with different sizes of the dots. This is best 
visualized if the dots are transparent:
Create random arrays with 100 values for x-points, y-points, colors and sizes:"""

import matplotlib.pyplot as plt
import numpy as np

x = np.random.randint(100, size=(100))
y = np.random.randint(100, size=(100))
colors = np.random.randint(100, size=(100))
sizes = 10 * np.random.randint(100, size=(100))

plt.scatter(x, y, c=colors, s=sizes, alpha=0.5, cmap='nipy_spectral')

plt.colorbar()

plt.show()

#Matplotlib Bars
"""Creating Bars
With Pyplot, you can use the bar() function to draw bar graphs:"""
import matplotlib.pyplot as plt
import numpy as np

x = np.array(["A","B","C","D"])
y = np.array([3,8,1,10])

plt.bar(x,y)
plt.show()

#Horizontal Bars
#If you want the bars to be displayed horizontally instead of vertically, use the barh() function:
#The bar() and barh() take the keyword argument color to set the color of the bars
import matplotlib.pyplot as plt
import numpy as np
x = np.array(["A", "B", "C", "D"])
y = np.array([3,5,7,1])
#The bar() takes the keyword argument width to set the width of the bars:
#The barh() takes the keyword argument height to set the height of the bars:
plt.barh(x, y, color = "limegreen", height = 0.4) # default height = 0.8
plt.show()

"""Histogram
A histogram is a graph showing frequency distributions.
It is a graph showing the number of observations within each given interval.
Example: Say you ask for the height of 250 people, you might end up with a histogram like this:"""

#In Matplotlib, we use the hist() function to create histograms.
"""For simplicity we use NumPy to randomly generate an array with 250 values, where 
the values will concentrate around 170, and the standard deviation is 10"""

import matplotlib.pyplot as plt
import numpy as np
x = np.random.normal(170, 10, 250) #(values will concentrate around 170,std deviation 10,generate an array with 250 values)

plt.hist(x)
plt.show()

"""
Matplotlib Pie Charts

Creating Pie Charts
With Pyplot, you can use the pie() function to draw pie charts:"""
import matplotlib.pyplot as plt
import numpy as np

y = np.array([35,25,12,13])
plt.pie(y)
plt.show()

"""Labels
Add labels to the pie chart with the labels parameter.
The labels parameter must be an array with one label for each wedge:"""
import matplotlib.pyplot as plt
import numpy as np
y = np.array([35, 30, 20, 10, 5])
mylabels = ["apples", "banana", "mango", "orange", "grapes"]
plt.pie(y, labels = mylabels, startangle = 90) #bydefault startangle = 0; 
plt.show()

"""Explode
Maybe you want one of the wedges to stand out? The explode parameter allows you to do that.
The explode parameter, if specified, and not None, must be an array with one value for each wedge."""
#Pull the "Apples" wedge 0.2 from the center of the pie:
#Add a shadow to the pie chart by setting the shadows parameter to True:
import matplotlib.pyplot as plt
import numpy as np

y = np.array([35, 25, 25, 15])
mylabels = ["Apples", "Bananas", "Cherries", "Dates"]
myexplode = [0.1, 0, 0, 0]
mycolors = ["limegreen", "hotpink", "magenta", "#4CAF50"]
plt.pie(y, labels = mylabels, colors = mycolors, explode = myexplode, shadow = True)
plt.show()

"""Legend
To add a list of explanation for each wedge, use the legend() function:"""

import matplotlib.pyplot as plt
import numpy as np
y = np.array([35, 25, 25, 15])
mylabels = ["Apples", "Bananas", "Cherries", "Dates"]
plt.pie(y, labels = mylabels)
plt.legend("Four Fruits:")  #Add a legend with a header:
plt.show() 

#Subplots (Multiple Plots in One Figure)
categories = ['A', 'B', 'C', 'D']
values = [10, 20, 15, 25]
fig, axs = plt.subplots(2, 2, figsize=(8, 8))
data = np.random.randn(1000)  # 1000 random values
axs[0, 0].plot(x, y, 'r')  # First subplot
axs[0, 1].scatter(x, y, color='blue')  # Second subplot
axs[1, 0].bar(categories, values, color='green')  # Third subplot
axs[1, 1].hist(data, bins=20, color='orange')  # Fourth subplot

plt.show()

"""Seaborn (Advanced Statistical Visualizations)
Seaborn is built on top of Matplotlib, making plots more beautiful & insightful."""
import seaborn as sns
import pandas as pd

#Line Plot Example
tips = sns.load_dataset("tips")  # Preloaded dataset in Seaborn
sns.lineplot(x="total_bill", y="tip", data=tips, hue="sex", style="sex")
plt.title("Line Plot Example")
plt.show()
# Used to check trends in data 📉

# Scatter Plot with Regression Line
sns.regplot(x="total_bill", y="tips" data=tips)
plt.title("Regression plot example")
plt.show()
#Used in ML to check relationships between variables 📌

#Box Plot (Detecting Outliers)
#Used in Data Cleaning to detect outliers 📊
sns.boxplot(x="day", y="total_bill", data=tips)
plt.title("Box plot Example")
plt.show()

#Heatmap (Correlation Matrix)
corr = tips.corr() ## Compute correlation matrix
sns.heatmap(corr, annot=True, cmap="coolwarm", linewidth="o.5")
plt.show()

"""The annot=True parameter adds the correlation coefficient values to each cell in the heatmap, 
making it easier to read the exact correlation values. The cmap="coolwarm" parameter specifies the
color map to be used, which in this case is a gradient from cool to warm colors. The linewidth="o.5" 
parameter sets the width of the lines that divide the cells in the heatmap, but there is a typo here 
as well; it should be linewidths=0.5."""

#Pairplot (Multi-variable Scatterplots)
sns.pairplot(tips, hue="sex")
plt.show()
"""hue="sex" parameter is used to differentiate data points based on the sex variable 
in the dataset. This means that data points will be colored differently based on the 
value of the sex variable, allowing for easy visual comparison between different groups."""
