# Deep-Learing group hw

**target**

- Use MAMEToolkit to train a fighter to play in the Street Fighter III (Asian) ROMS_id = sfiiia
- Use A3C method to train it

**QuickStart**
```sh
git clone https://github.com/JinhuaSu/sfiiia-a3c.git
cd src
python main.py
```

**DDL**

2019-12-25 have a presetation

**process**

| DDL | task | state |
| :--: | :--: | :--: |
| 2019-12-10 | read a3c code | done |
| 2019-12-10 | make the game like gym.env | done |
| 2019-12-15 | first demo | done |

**urgent**

*find a game and download it. detail can be seen in the ./src/env/search.txt*
done in 2019-12-10:01:09


**to-do-list**

> a small cake which can be finished independently

| importance | task | difficulty | who |
| :--: | :--: | :--: | :--: |
| xx | use tensorboard draw the model graph of a3c | xx |  |
| x | test another gym game in this code | x | done with Pong-v0 |
| xxx | use gym class and make the sfiiia as a standard gym | xxx | done with sfenvironment |
| xxxx | most 
| xxx | write a class py for action(hierarchical) | xx |  | 
| xxxx | write a class py for reward(hierarchical) | xx |  |
| x | make the windows can be pretty show | xx |  |
| xxx | write a python can select different mode so that I can play with the our fighter | xx |  |

**the story of king of fighter**

- king I
    + problem: no super act, no memory, small
    + improve: 20M -> 1.2G, add super act to action list
- king II
    + problem: too much defence
    + improve method: change the reward mode, add subprocess and train for 9 hours
- king III
    + a mature version: can fight 8 difficulty computer
    + problem: not best hyperparameter
    + improve: test and adjust the hyperparameter, 
- king IV
    + remove the unnecessary color context, and improve charater generalization
    
        - 

