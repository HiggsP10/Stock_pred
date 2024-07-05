import os
import pandas as pd
import nltk
from nltk.corpus import stopwords
def Preprocess_Tweets(data):
		
	data['Text'] = data['Text'].str.lower()

	## FIX HYPERLINKS
	data['Text'] = data['Text'].replace(r'https?:\/\/.*[\r\n]*', ' ',regex=True)
	data['Text'] = data['Text'].replace(r'www.*[\r\n]*', ' ',regex=True)
	data['Text'] = data['Text'].str.replace('https', '', regex=False)


	## FIX INDIVIDUAL SYMBOLS 
	data['Text'] = data['Text'].str.replace(': ', ' ', regex=False)
	data['Text'] = data['Text'].str.replace(', ', ' ', regex=False)
	data['Text'] = data['Text'].str.replace('. ', ' ', regex=False)
	data['Text'] = data['Text'].str.replace('[;\n~]', ' ', regex=True)

	data['Text'] = data['Text'].str.replace("[]'â€¦*™|]", '', regex=True)
	data['Text'] = data['Text'].str.replace('[[()!?"]', '', regex=True)
	data['Text'] = data['Text'].str.replace('_', '', regex=False)
	data['Text'] = data['Text'].str.replace('w/', ' with ', regex=False)
	data['Text'] = data['Text'].str.replace('f/', ' for ', regex=False)


	## FIX EMOJIS
	data['Text'] = data['Text'].str.replace(':)', '', regex=False)
	data['Text'] = data['Text'].str.replace(':-)', '', regex=False)
	data['Text'] = data['Text'].str.replace(':(', '', regex=False)
	data['Text'] = data['Text'].str.replace(':-(', '', regex=False)
	data['Text'] = data['Text'].str.replace('0_o', '', regex=False)
	data['Text'] = data['Text'].str.replace(';)', '', regex=False)
	data['Text'] = data['Text'].str.replace('=^.^=', '', regex=False)


	## FIX % SYMBOL
	data['Text'] = data['Text'].str.replace('%', ' percent ', regex=False)


	## FIX & SYMBOL
	data['Text'] = data['Text'].str.replace(' & ', ' and ', regex=False)
	data['Text'] = data['Text'].str.replace('&amp', ' and ', regex=False)
	data['Text'] = data['Text'].str.replace('&gt', ' greater than ', regex=False)
	data['Text'] = data['Text'].str.replace('cup&handle', 'cup and handle', regex=False)
	data['Text'] = data['Text'].str.replace('c&h', 'cup and handle', regex=False)
	data['Text'] = data['Text'].str.replace('head&shoulders', 'head and shoulders', regex=False)
	data['Text'] = data['Text'].str.replace('h&s', 'head and shoulders', regex=False)
	data['Text'] = data['Text'].str.replace('point&figure', 'point and figure', regex=False)
	data['Text'] = data['Text'].str.replace('p&f', 'point and figure', regex=False)
	data['Text'] = data['Text'].str.replace('s&p', 'SP500', regex=False)
	data['Text'] = data['Text'].str.replace('q&a', 'question and answer', regex=False)
	data['Text'] = data['Text'].str.replace('&', ' and ', regex=False)


	## FIX USER TAGS AND HASTAGS
	data['Text'] = data['Text'].str.replace('@[a-z0-9]+', '', regex=True)
	data['Text'] = data['Text'].str.replace('#[a-z0-9]+', '', regex=True)
	data['Text'] = data['Text'].str.replace('@', '', regex=False)
	data['Text'] = data['Text'].str.replace('#', '', regex=False)
	   
		
	## FIX EMBEDDED COMMAS AND PERIODS    
	data['Text'] = data['Text'].replace(r'([a-z]),([a-z])', r'\1 \2', regex=True)
	data['Text'] = data['Text'].replace(r'([0-9]),([0-9])', r'\1\2', regex=True)
	data['Text'] = data['Text'].replace(r'([0-9])[+]+', r'\1 ', regex=True)
	data['Text'] = data['Text'].str.replace(',', '', regex=False)
	data['Text'] = data['Text'].str.replace('u.s.', ' us ', regex=False)
	data['Text'] = data['Text'].str.replace('\.{2,}', ' ', regex=True)
	data['Text'] = data['Text'].replace(r'([a-z])\.([a-z])', r'\1 \2', regex=True)
	data['Text'] = data['Text'].str.replace('pdating', 'updating', regex=False) 
	data['Text'] = data['Text'].replace(r'([a-z])\.', r'\1 ', regex=True)
	data['Text'] = data['Text'].replace(r'\.([a-z])', r' \1', regex=True)
	data['Text'] = data['Text'].str.replace(' . ', ' ', regex=False)
		

	## FIX + SYMBOL
	data['Text'] = data['Text'].replace(r'[+]([0-9])', r'positive \1', regex=True)
	data['Text'] = data['Text'].str.replace('c+h', 'cup and handle', regex=False)
	data['Text'] = data['Text'].str.replace('h+s', 'head and shoulders', regex=False)
	data['Text'] = data['Text'].str.replace('cup+handle', 'cup and handle', regex=False)
	data['Text'] = data['Text'].str.replace(' + ', ' and ', regex=False)
	data['Text'] = data['Text'].str.replace('+ ', ' ', regex=False)
	data['Text'] = data['Text'].replace(r'([a-z])[+]([a-z])', r'\1 and \2', regex=True)
	data['Text'] = data['Text'].str.replace('+', '', regex=False)



		
	## FIX - SYMBOL
	data['Text'] = data['Text'].replace(r'([a-z])[-]+([a-z])', r'\1 \2', regex=True)
	data['Text'] = data['Text'].replace(r'([a-z]) - ([a-z])', r'\1 to \2', regex=True)
	data['Text'] = data['Text'].replace(r'([0-9]) -([0-9\.])', r'\1 to \2', regex=True)
	data['Text'] = data['Text'].replace(r' [-]([0-9])', r' negative \1', regex=True)
	data['Text'] = data['Text'].replace(r'([0-9])-([0-9\.])', r'\1 to \2', regex=True)
	data['Text'] = data['Text'].replace(r'([0-9]) - ([0-9\.])', r'\1 to \2', regex=True)
	data['Text'] = data['Text'].replace(r'([0-9a-z])-([0-9a-z])', r'\1 \2', regex=True)
	data['Text'] = data['Text'].str.replace('[-]+[>]', ' ', regex=True)
	data['Text'] = data['Text'].str.replace(' [-]+ ', ' ', regex=True)
	data['Text'] = data['Text'].str.replace('-', ' ', regex=False)



	## FIX $ SYMBOL
	data['Text'] = data['Text'].str.replace('[$][0-9\.]', ' dollars ', regex=True)
	data['Text'] = data['Text'].str.replace('$', '', regex=False)


	## FIX = SYMBOL
	data['Text'] = data['Text'].str.replace('=', ' equals ', regex=False)

		
	## FIX / SYMBOL
	data['Text'] = data['Text'].str.replace('b/c', ' because ', regex=False)
	data['Text'] = data['Text'].str.replace('b/out', ' break out ', regex=False)
	data['Text'] = data['Text'].str.replace('b/o', ' break out ', regex=False)
	data['Text'] = data['Text'].str.replace('p/e', ' pe ratio ', regex=False)
	data['Text'] = data['Text'].str.replace(' [/]+ ', ' ', regex=True)
	data['Text'] = data['Text'].str.replace(' 1/2 ', ' .5 ', regex=False)
	data['Text'] = data['Text'].str.replace(' 1/4 ', ' .25 ', regex=False)
	data['Text'] = data['Text'].str.replace(' 3/4 ', ' .75 ', regex=False)
	data['Text'] = data['Text'].str.replace(' 1/3 ', ' .3 ', regex=False)
	data['Text'] = data['Text'].str.replace(' 2/3 ', ' .6 ', regex=False)

	data['Text'] = data['Text'].str.replace('[/]{2,}', ' ', regex=True)
	data['Text'] = data['Text'].replace(r'([a-z])/([a-z])', r'\1 and \2', regex=True)
	data['Text'] = data['Text'].str.replace('[0-9]+/[0-9]+/[0-9]+', '', regex=True)
	data['Text'] = data['Text'].replace(r'([0-9]{3,})/([0-9\.]{2,})', r'\1 to \2', regex=True)
	data['Text'] = data['Text'].replace(r'([0-9]{2,})/([0-9\.]{3,})', r'\1 to \2', regex=True)
	data['Text'] = data['Text'].str.replace('[a-z0-9]+/[a-z0-9]+', ' ', regex=True)

	data['Text'] = data['Text'].str.replace('/', '', regex=False)


	## FIX < > SYMBOLS
	data['Text'] = data['Text'].str.replace('[<]+ ', ' ', regex=True)
	data['Text'] = data['Text'].str.replace('<', ' less than ', regex=False)
	data['Text'] = data['Text'].str.replace(' [>]+', ' ', regex=True)
	data['Text'] = data['Text'].str.replace('>', ' greater than ', regex=False)


	## FIX : SYMBOL
	data['Text'] = data['Text'].str.replace('[0-9]+:[0-9]+am', ' ', regex=True)
	data['Text'] = data['Text'].str.replace('[0-9]+:[0-9]', ' ', regex=True)
	data['Text'] = data['Text'].str.replace(':', ' ', regex=False)


	## FIX UNITS
	data['Text'] = data['Text'].str.replace('user ', ' ', regex=False)

	data['Text'] = data['Text'].replace(r'([0-9]+)dma', r'\1 displaced moving average ', regex=True)
	data['Text'] = data['Text'].replace(r'dma([0-9]+)', r'\1 displaced moving average ', regex=True)
	data['Text'] = data['Text'].replace(r'([0-9]+)sma', r'\1 simple moving average ', regex=True)
	data['Text'] = data['Text'].replace(r'sma([0-9]+)', r'\1 simple moving average ', regex=True)
	data['Text'] = data['Text'].replace(r'([0-9]+)ema', r'\1 expontential moving average ', regex=True)
	data['Text'] = data['Text'].replace(r'ema([0-9]+)', r'\1 expontential moving average ', regex=True)
	data['Text'] = data['Text'].replace(r'([0-9]+)ma', r'\1 moving average ', regex=True)
	data['Text'] = data['Text'].replace(r'ma([0-9]+)', r'\1 moving average ', regex=True)

	data['Text'] = data['Text'].replace(r'([0-9])mos', r'\1 months ', regex=True)
	data['Text'] = data['Text'].replace(r'([0-9])minute', r'\1 minute ', regex=True)
	data['Text'] = data['Text'].replace(r'([0-9])minutes', r'\1 minutes ', regex=True)
	data['Text'] = data['Text'].replace(r'([0-9])min', r'\1 minute ', regex=True)
	data['Text'] = data['Text'].replace(r'([0-9])mins', r'\1 minutes ', regex=True)
	data['Text'] = data['Text'].replace(r'([0-9])day', r'\1 day ', regex=True)
	data['Text'] = data['Text'].replace(r'([0-9])days', r'\1 days ', regex=True)
	data['Text'] = data['Text'].replace(r'([0-9])wk', r'\1 week ', regex=True)
	data['Text'] = data['Text'].str.replace(' wk ', ' week ', regex=False)
	data['Text'] = data['Text'].str.replace(' wknd ', ' weekend ', regex=False)
	data['Text'] = data['Text'].replace(r'([0-9])wks', r'\1 weeks ', regex=True)
	data['Text'] = data['Text'].replace(r'([0-9])hours', r'\1 hours ', regex=True)
	data['Text'] = data['Text'].replace(r'([0-9])hour', r'\1 hour ', regex=True)
	data['Text'] = data['Text'].replace(r'([0-9])yr', r'\1 year ', regex=True)
	data['Text'] = data['Text'].replace(r'([0-9])yrs', r'\1 years ', regex=True)
	data['Text'] = data['Text'].str.replace(' yr', ' year ', regex=False)
	data['Text'] = data['Text'].replace(r'([0-9])am', r'\1 am ', regex=True)
	data['Text'] = data['Text'].replace(r'([0-9])pm', r'\1 pm ', regex=True)

	data['Text'] = data['Text'].replace(r'([0-9])est', r'\1 ', regex=True)
	data['Text'] = data['Text'].replace(r'([0-9])ish', r'\1 ', regex=True)
	data['Text'] = data['Text'].replace(r'([0-9 ])pts', r'\1 points ', regex=True)
	data['Text'] = data['Text'].replace(r'([0-9])x', r'\1 times ', regex=True)
	data['Text'] = data['Text'].replace(r'([0-9])th', r'\1 ', regex=True)
	data['Text'] = data['Text'].replace(r'([0-9])rd', r'\1 ', regex=True)
	data['Text'] = data['Text'].replace(r'([0-9])st', r'\1 ', regex=True)
	data['Text'] = data['Text'].replace(r'([0-9])nd', r'\1 ', regex=True)

	data['Text'] = data['Text'].str.replace('mrkt', 'market', regex=False)
	data['Text'] = data['Text'].str.replace(' vol ', ' volume ', regex=False)
	data['Text'] = data['Text'].str.replace(' ptrend', ' positive trend ', regex=False)
	data['Text'] = data['Text'].str.replace(' ppl', ' people ', regex=False)
	data['Text'] = data['Text'].str.replace(' pts', ' points ', regex=False)
	data['Text'] = data['Text'].str.replace(' pt', ' point ', regex=False)
	data['Text'] = data['Text'].str.replace(' l(ol){1,}', ' laugh ', regex=True)
	data['Text'] = data['Text'].str.replace('imho', ' in my opinion ', regex=True)
	data['Text'] = data['Text'].str.replace('prev ', 'previous ', regex=True)


	data['Text'] = data['Text'].str.replace(' 1q', ' first quarter ', regex=False)
	data['Text'] = data['Text'].str.replace(' 2q', ' second quarter ', regex=False)
	data['Text'] = data['Text'].str.replace(' 3q', ' third quarter ', regex=False)
	data['Text'] = data['Text'].str.replace(' 4q', ' fourth quarter ', regex=False)
	data['Text'] = data['Text'].str.replace(' q1', ' first quarter ', regex=False)
	data['Text'] = data['Text'].str.replace(' q2', ' second quarter ', regex=False)
	data['Text'] = data['Text'].str.replace(' q3', ' third quarter ', regex=False)
	data['Text'] = data['Text'].str.replace(' q4', ' fourth quarter ', regex=False)
	data['Text'] = data['Text'].str.replace(' 10q ', ' form 10 ', regex=False)

	data['Text'] = data['Text'].replace(r'([0-9])million', r'\1 million ', regex=True)
	data['Text'] = data['Text'].replace(r'([0-9])mil', r'\1 million ', regex=True)
	data['Text'] = data['Text'].str.replace(' mil ', ' million ', regex=False)
	data['Text'] = data['Text'].replace(r'([0-9])billion', r'\1 billion ', regex=True)
	data['Text'] = data['Text'].replace(r'([0-9])cents', r'\1 cents ', regex=True)

	data['Text'] = data['Text'].replace(r'([0-9])3d', r'\1 3 dimensional ', regex=True)
	data['Text'] = data['Text'].replace(r'([0-9])gb', r'\1 3 gigabytes ', regex=True)



	data['Text'] = data['Text'].replace(r'([0-9])c', r'\1 calls ', regex=True)
	data['Text'] = data['Text'].replace(r'([0-9])y', r'\1 year ', regex=True)
	data['Text'] = data['Text'].replace(r'([0-9])p', r'\1 puts ', regex=True)
	data['Text'] = data['Text'].replace(r'([0-9])d', r'\1 days ', regex=True)
	data['Text'] = data['Text'].replace(r'([0-9])h', r'\1 hour ', regex=True)
	data['Text'] = data['Text'].replace(r'([0-9])s', r'\1 ', regex=True)
	data['Text'] = data['Text'].replace(r'([0-9])k1', r'\1 thousand ', regex=True)
	data['Text'] = data['Text'].replace(r'([0-9])k', r'\1 thousand ', regex=True)
	data['Text'] = data['Text'].replace(r'([0-9])m', r'\1 million ', regex=True)
	data['Text'] = data['Text'].replace(r'([0-9])b', r'\1 billion ', regex=True)

		
	data['Text'] = data['Text'].replace(r'([0-9])([a-z])', r'\1 \2', regex=True)

	## FIX EXTRA SPACES AND ENDING PUNCTUATION
	data['Text'] = data['Text'].str.replace(' +', ' ', regex=True)
	data['Text'] = data['Text'].str.strip(' .!?,)(:-')
	#Clear up meaningless words for NB
	Stop = set([s.replace("'", '') for s in stopwords.words('english') if s not in ['not', 'up', 'down', 'above', 'below', 'under', 'against','no','shouldnt']])
	data['Text'] = data['Text'].apply(lambda s: " ".join([word for word in s.split() if word not in Stop]))
	data['Text'] = data['Text'].str.strip()

	return data