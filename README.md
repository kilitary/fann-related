For example:


FANN networks [+ forex tasks that was applied to it using this software]
This is just command line tools for FANN. No gui, but many options and real-time switching not only temps/decoys or any other params (u think) affect this. 
+auto-tuning may train network starting using RPROP or QUICKPROP, then when hit ratio did not changes for example last 25 epochs (stupid ceil counts to examine in train, the will range in 6-53 epochs or any other range u cant imagine coz.) will increase/decrease some parameter it chooses. and u have full control of jittering, which is very important if u want to apply network to data u did not see. 
with this express-coded "software" u can create (sparse,connected or other FANN type network) ,train using all the methods FANN have,run with different configurations,analyzing where it is BAD TRAIN DATA, that takes big time to understand, especially for ppl that did not read all these super-algebraic pdfs with anylyzing dozen of "crazy-good" documents on the internet while every science-writer adding their situation to problem that it follow to be reached in its train data, it starting temperatures(!! one document say simulated annealing starts at millions of CELCIUS and decreases by 0.002 or smtn steps, while other found that cooling works too (??) , not ranges for examples. 
u also can view data (if u have eyes or some other indicators) in separate command line utilities, compiled by mingw (but vs/other can compile it too, u just need FANN for u'r platform, and i think intel compilator is best choice, i not used TBB but as i did reached my goals not using it, adding TBB will not increase HIT rate, but can do faster bruteforcing(mutation) of all possible logic lines in train sheme if u ever hopes in mutation success of finding reach faster than communicating all the info u have and normalize it, like the 49% percent of scientisc (or ppl that wrote all that info i was crazy half a year) says about range -1.0 - 1.0, the other 49% says u can use plain not normalized params, well that interesing and shows that the ppl (maybe) that fucking do something in NN or AI development flyes in random galactics, because it is not proofed? i did find %50 vs %50, but really that info should be calcutable i think? what is nn? it is not a skynet computer that calculated in 0.047 ms that .... that secret info is secretandsorry i cant continue, cya. my FANN life helicopter waits me to proceed to another area, to continue what i want to enchance in this situation with this annoing pdf's every of i read did ..). 
+working(implemented) simulated annealing train and working auto-tune parameters, for heating or colding - which FANNtool/others does  not have (kill me if u know that software)
many settings u can use through cmdline to set up temperature/decoys/direction (colding or heat) 
RPROP/QUICKPROP also have auto-tunes. i cant say my approach to auto-tune that params is best, but it gives 83% hit rate and combining earlier train with noise data and jittering. 
software can log data for gnuplot program and updates in real-time, you can pause train, change any params using keyboard and continue, while viewing the process as it goes. or u can not pause, but change all too =)
+some stupid utilities that bruteforce random network configurations using all possible activation functions and stepness range.
ppl are free to do anything with that.
+some "mutation" while currently it uses english alphabet as a reproducable approach to catch the program when you see success.
software sees this too, but just using standard mathematic method, while visual information encanches this process, and u can
(or u not can i do not know about u) view moments when u can fastly switch from RPROP to SA or other scheme.
yes i know noone does that, as i readed tons of pdfs. but, SA is heating algorithm, now some pdf's says about combining even colding with using for example genetic algorithms (that i did not understand for the moment)
really some things like starting using QUICKPROP will drive you to fast minimum, where SA (if you are lucky) can go further.
+simple cascaded training, i did not prefer cascaded train (i do not remember why, because dont need that).
+list network connects
+train data classifier, it can use two methods, dividing classes so every class will have same number of samples, or use vector-controlled training, when u know u need more reject class samples. 
i will not write all these momments, but really u will got all u need to train any network
(except some random coredumps, but this is not main problem, the network saves net when it reaches some point,
and doing the copy software continues reaching success until (if u're will setup this) stopps because of not HIT ratio. 
this software is builded more on HIT ratio rather than decreasing MSE, its because in digital world i do not need 0.3 or 0.8 as exit from network. 
most of users need to implement basic logic (boolean tasks) when network can say YES or NO. i do not know medicine problems, where it is needed, or alike.
or if u use reject class - nothing at all (what shows u're have some error in training data or parameters.
ever increasness in HIT ratio makes dump of network to std FANN format file.


none license, no moneyback, no help (write me messages, i am still working on this, even not on computer)


thanks for some people wich helped me doing big HIT Ratio (#ircforex) (83% of hit rate is best i can reached, but i seed even 95% but.. hm some things did not synced in my algorithm with my work list and big hit ratio was killed by other things like technical analysis which i EVEN NOT USED, cos i am normal human and the program can do that.
well, 83% hit rate in AA rockets i think is not bad value, but the rocket has fuel, and when it finally ends, all the AI/other algorithms goes to background, of course if rocket can attack fighter and take it fuel, like to proceed to more realiable target for example, a bigger airplane and so on....
in forex world, i have been occupied by stupid media-agencys that linked some troyan info, which as i investigated then are mass-discuted for example on ircforex.com IRC channels. reproducing some that info, multiplying going the thing i more hate - the program cant more be assigned only to my algorithm, i should change the things i used earlier to write the PROGRAM (which consist not only by training the net, u should enter values from indicators that technically in current (forex) world are just copy of logic ups and downs in range, indicators (like real world indicators, thank world) to 
, like the data in real world is not only changing in range -1.0 - 1.0 but as (as NN "still" cant do that problem by using fast-mutation or hybrid genetic+SA algorithm for which i too lerned bad in the school to implement just in "some time")
the logic can changes just when u see new train data, because the yesterday we know not tomorrow, and theone doing for example media/politics/4th other ppl, that can to do approaching when modifying of the prices is not technically-reverse-engeneered like using super indicators that comes with auto-trade bots for example (the slow rubicon for example "gives view of some visual informatin that will(can) predict the trend, or working CRYSIS (thx all) , (slightly modified - can make money yes, if u have 16k$ for example and can make multiple orders in both directions and close lossy when u have more profitable orders. ..(i talk about forex, coz it is only really-investigated in express mode using last techniqs for that)



all that data is investigated by me and for me for other ppl that will enter NN-technologies having no idea at all how it works.
but, if you're still not died reading this, look i have really good SA.
.
 some ppl (forex traders, that i "seeed" in my MIRC when no connections are stable to that irc network, surprised me) while stopped me (of course noone stopped me, i just dont want to look at NN i create that cant predict the information because stupid humans are different aand not in digital formation, sleeps in random locations so when one sleeps and not making money another forcing to exploit their situatio) 
 aaaaand to reach what i want to do using this networks i entered some other businessesss and programs and investigations! (for example virus detecting in earlier state (which u can see in AVZ with not changing results of networks for different files, looks like the work that is not interested to continue. . that was investigated by me, and no - u cant do that (u can on quantum, maybe, or on aliens super-computer which i think did not operate with resistance) . maybe on quantum computers my maybe is not indeterminent, but i trained the network that using first 1024 (for ex) first bytes of file can say it extension. this was done using reject class too, because what i can to say about that gaps in the file? padding and other things, like export tables(in this example) is VERY STATIC to detect that this file is not a sample network seeded when trained.
 so many files (for example the TXT files u know the ppl writes alphabet using some really-predictable algorithm, coz the word consist from micro-elements and other things like unrouted lingvistics. so, yes i can say i reached 70% success of HIT rate for detecting 
 jpeg (haha) pictures, or EXE's, f course beacuse they have PE32 format and are not too different from file to file. but i stopped on this, because that did not make moneys for me so i am not interrested for now. , while i follows only financial intereset in i cover different aspects of information about AI(no,no artificial intelligence) that descriebes the situation with AI/NN developing on this planet, i found that it does not have real sources of problems so the ppl can test&drive NN/AI further, following to create predict-machine, but not the AI/NN(when using in AI) YEs, yes yes i know i am idiot with bad matemathic knowledges, but the digits is simple for me to understand when i use all that (ppl created, what ppl wrote that algo like hybrid cooling of SA?) looks like some mathematican uses not only digits but euristic approaches.
 
 fuck, some ppl will blame i wrote such text, but this text is that i use every day to easier my life. dont ask me how FANN command line tool software changes the lifes of pppl (i am not talking about bad sex robotics furthers of this aspect, thx jap), some ppl have that info. 
 
 that.
