# Proposal

## What will (likely) be the title of your project?

Sports Graphics Suite

## In just a sentence or two, summarize your project. (E.g., "A website that lets you buy and sell stocks.")

A suite of programs to create and display live, professional, detailed, and comprehensive sports graphics.

## In a paragraph or more, detail your project. What will your software do? What features will it have? How will it be executed?

Scoreboard - Scoreboard that can be adapted for just about any sport, with accurate data from the actual score system. HTML/CSS/Javascript/Websockets. WIll have a controller that can update certain score elements or override other things like the OCR program.

Stats - An additional controller that will allow one or multiple operators to record game statstics that can be used on the scoreboard. The main operator will be able to select graphics to show them at certain times or have them on the screen continuously. Will be HTML based, and adaptable for any sport. Import rosters as CSV or JSON, and add statistics that can be measured for each player, or for certain player positions that will be added. Will be able to have presets, but will also be easily expandable.

OCR - Python program that will recognize digits on any scoreboard, as long as a camera is pointed at it. Will be used especially for time based events like game clocks, shot clocks, and race times, but also adaptable for other score items so that a broadcast can be run without a graphics operator. Using OpenCV, you can slice the image to refer to certain subsections, and perform certain operations to make the image into a small image that can be compared to a set of reference digits. The program will also be able to recognize different fonts (e.g. the traditional [seven segment digits](https://t3.ftcdn.net/jpg/02/54/67/20/360_F_254672025_o9tbei4yh3d3zVNLTpuATQQZGrqwEfIs.jpg) or a [5x7 dot based lcd scoreboard](https://www.dafont.com/img/illustration/s/c/score_board.png))

## If planning to combine 1051's final project with another course's final project, with which other course? And which aspect(s) of your proposed project would relate to 1051, and which aspect(s) would relate to the other course?

I work at my alma mater as a media/livestreaming coordiantor. These graphics will be used for our broadcasts. I also plan to show these graphics as part of a portfolio for potential internships. 

## In the world of software, most everything takes longer to implement than you expect. And so it's not uncommon to accomplish less in a fixed amount of time than you hope.

### In a sentence (or list of features), define a GOOD outcome for your final project. I.e., what WILL you accomplish no matter what?

Scoreboard - HTML/CSS/Javascript based graphics that can be controlled via WebSockets

### In a sentence (or list of features), define a BETTER outcome for your final project. I.e., what do you THINK you can accomplish before the final project's deadline?

Basic OCR - Python program to recognize characters on a scoreboard when a camera is pointed at it. Get coordinates, threshold (black and white), scale them, create reference digits, read the clock

### In a sentence (or list of features), define a BEST outcome for your final project. I.e., what do you HOPE to accomplish before the final project's deadline?

Stats - Program for an operator to plug in stats that can be used with the advanced scoreboard
Advanced Scoreboard - Better contorller, scoreboard with more animations, features, changable for each sport, can make adjustments easily
Best OCR - Better software, more features, multiple scoreboard fonts, recognize other scoreboard numbers, possibly recognize text from natatorium scoreboard, track color changes (possibly package the software onto a compute stick, NUC, or raspberry pi so that it can be attached to a camera that can be anywhere)

## In a paragraph or more, outline your next steps. What new skills will you need to acquire? What topics will you need to research? If working with one of two classmates, who will do what?

I am already familiar with most of the HTML-based things, but I will definitely have to play around with Python and OpenCV. I'll have to figure out how to have a main program that can run the websockets program, the web server, the OCR program, etc.
