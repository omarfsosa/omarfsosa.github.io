---
layout: single
title:  "Life in Jupyter"
description: "5 actually useful tips"
date:   2021-03-30
mathjax: false
plotly: false
tags: [python, ipython, jupyter]
---

I use jupyter notebooks in my work every single day. Every. Day. Over time, I've learnt a few good tricks that make my life a little better. I hope you too will find them useful.


### 1. Drop the notebook, grab the lab.
If you're using Jupyter and you're not using Jupyter Lab, you're missing out. None of the cool kids are using `jupyter notebook` these days. If you're editing a single notebook it won't make much of a difference. But once you start having multiple notebooks in your project, you will need that side menu to navigate the file tree. Also, splitting your screen across multiple notebooks and the ability to drag and drop cells from one notebook to another is just priceless.

Just do `pip install jupyterlab` and from now on do `jupyter lab` instead of `jupyter notebook`. 

### 2. Multi-cursor.
Like with many editors, Jupyter lets you have multiple cursors within a single cell. Hold down your `cmd` key[^1] and click at the places where you'd like to add a cursor. Alternatively, holding `option` and dragging your mouse will create a block cursor. The multiple cursors work fine if you want to do some copy-pasting. For extra points, use `option` or `cmd` and your arrow keys to move your cursor quickly around the cell.

[^1]: I'm a macos user. Sorry not sorry.

![Comparison](/assets/images/blog-images/2021-03-30-life_in_jupyter/multicursor.gif)

This is what's happening in the example above:

1. I use `cmd + click` to add my cursor in the places where I wish to write `parse_dates`.
2. I write the desired code.
3. Hit `esc` to go back to a single cursor.
4. I hold `option` and drag my mouse 3 rows to create a block cursor. Then I prefix my variable names with `path_`.
5. I hold `shift + option` and hit the left arrow to take all three cursors to the start of the word while also selecting that part of the word.
6. I hit `cmd + c` to copy the selected text.
7. I hit my down arrow key until I get to the rows where I want to paste the text.
8. I hold `alt` while I hit my right arrow key to quickly move my cursor a few words forward.
9. I hit `cmd + v` to paste the text.
10. I hit `esc` twice. Once to go back to single cursor. A second time to stop editing the cell.

It might seem like a lot, but once you practice a bit it will become like a second nature and you won't even have to think about it.


### 3. Learn the keyboard shortcuts.

A piece of me dies whenever I see someone take their hand away from their keyboard, just to grab the mouse and click that `+` at the top for adding an extra cell. No, no. We don't do that. Yes, we are data scientist and coding is not our forte, but that doesn't mean we can't have a bit of style. Here, learn these and you'll look like a pro in no time:

- `a`: Add cell above.
- `b`: Add cell below.
- `ii`: Interrupt kernel.
- `00`: Restart the kernel.
- `j`: Move down one cell.
- `k`: Move up one cell.

Of course, to use any of the above commands you'll have to make sure your cursor is not active in the cell. To do so, just press the `esc` key. I personally have my `esc` remapped to my `caps` key, so that I can easily hit it with my left pinky. Who uses the `caps` key anyway?


### 4. Fetching the output of a given cell.
Here's the situation. After many iterations, the bit of code you wrote finally did what you wanted it to do ... but you didn't assign the output to any variable so you have to run the cell again. Except that it will take some time because it was a chunky piece of code. If only python allowed you to refer to the last output ...

Here. You're welcome.

![Comparison](/assets/images/blog-images/2021-03-30-life_in_jupyter/underscore.png)


Except that, maybe, you've already used the underscore to iterate over some obscure for-loop so the above trick doesn't quite work for you. Fear not:

![Comparison](/assets/images/blog-images/2021-03-30-life_in_jupyter/out.png)


### 5. High resolution plots.

If you're already using `%matplotlib inline`, then simply add the magic `%config InlineBackend.figure_format = "retina"` right after that.

Low Resolution             |  High Resolution
:-------------------------:|:-------------------------:
![](/assets/images/blog-images/2021-03-30-life_in_jupyter/lores.png)  |  ![](/assets/images/blog-images/2021-03-30-life_in_jupyter/hires.png)


Finally, make sure you pretend you've always known these things.

