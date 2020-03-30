# 2020 Inui-Suzuki Lab "100-knock" NLP exercises

You will create natural language processing programs and review each other's codes.
There is no single right answer, so try to solve it in any possible way. 

## Questions

[100-knock questions in English](https://github.com/cl-tohoku/nlp100-questions)

## Format

- The TA/mentor will specify which questions to solve until next time
- Students should prepare solutions for the specified questions
  - Even if you don't successfully solve a question, leave your attempt visible as-is
  - Please do not leave it at "I didn't know/I couldn't solve it"
- At each session, we will discuss the contents of each other's code.


## Setting up

```bash
git clone 'git@github.com:cl-tohoku/100knock-2020.git'
cd 100knock-2020
git checkout -b [username]
mkdir [username]
git commit --allow-empty -m "Initial commit"
git push -u origin [username]
```

After that, place your code under your personal directory.

## To do every time you push your code

(after `cd` to 100knock-2020)

```bash
git add [username]
git commit -m "A cool comment (e.g., what you changed)"
git push
```

## Structure of personal directory

Please structure your directory as follows:

```plain
ryo-t/
  ├ chapter01/
  │   └ ryo-t_ch01.ipynb <- your name should be included
  ├ chapter02/
  │   ├ data/            <- Place the data to be processed (before processing)
  │   │   └ hightemp.txt
  │   ├ src/             <- If you use any scripts, place them here
  │   │   ├ q014.py
  │   │   ├ q015.py
  │   │   └ q016.py
  │   ├ work/            <- Place intermediate products of processing here
  │   │   ├ col1.txt
  │   │   ├ col2.txt
  │   │   ├ xaa
  │   │   ├ xab
  │   │   ├ xac
  │   │   ├ xad
  │   │   └ xae
  │   └ ryo-t_ch02.ipynb
  ├ chapter03/
  .
  .
  .
```

`data` and `work` are excluded from git source control (will be ignored).

## TA/Mentor's job

Merge the participants' branches to the master branch before starting. Specifically, execute the following commands:

```bash
scripts/merge_all_into_master.sh  # if a conflict occurs, properly eliminate it
git push
```
