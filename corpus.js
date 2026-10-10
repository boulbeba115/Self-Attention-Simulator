    /* =====================================================================
       TRAINING TEXT — the model's whole world
       Hand-written sentences, small fact tables that are expanded into
       sentences, and a large sample of short stories from corpus-stories.js.
       One sentence per line. script.js counts the phrases in it and learns
       its word vectors from it.
       ===================================================================== */
    const CORPUS_TEXT = (() => {
      const out = [];
      const lines = s => s.trim().split('\n').map(x => x.trim()).filter(Boolean);
      const an = w => (/^[aeiou]/.test(w) ? 'an ' : 'a ') + w;
      const add = (...s) => out.push(...s);

      /* ---------- everyday talk ---------- */
      add(...lines(`
hello how are you
hello how are you today
hello how are you doing
hello how is it going
hello my friend
hello world
hello everyone
hello there
hello and welcome
hello nice to meet you
hi how are you
hi there my friend
hi everyone
hey how is it going
good morning everyone
good morning how are you
good morning my friend
good morning and welcome
good afternoon everyone
good evening my friend
good evening and welcome
good night and sweet dreams
good night my friend
good luck with your exam
good job my friend
how are you today
how are you doing today
how are you feeling today
how was your day
how was your weekend
how is your family
how old are you
how much does it cost
how much is this
how many people are there
how do you do
how do you say that in english
how can i help you
how long does it take
how far is the station
i am fine thank you
i am fine and you
i am doing well thank you
i am very well thank you
i am happy to see you
i am happy to help you
i am glad to meet you
i am sorry i am late
i am sorry about that
i am tired today
i am hungry
i am thirsty
i am ready
i am not sure
i am at home
i am at work
i am on my way
i am a student
i am a teacher
i am from tunisia
i am learning english
i am looking for the station
i am proud of you
thank you very much
thank you so much
thank you for your help
thank you for your time
thank you for coming
thank you my friend
you are welcome
you are right
you are very kind
you are my best friend
you are doing a great job
nice to meet you
nice to meet you too
nice to see you again
see you tomorrow
see you later my friend
see you soon
see you next week
have a nice day
have a good day
have a good night
have a great weekend
have a safe trip
what is your name
what is your favourite colour
what is your favourite food
what is the weather like today
what is this
what is that
what is the time
what time is it
what are you doing
what are you doing today
what do you want to eat
what do you think about it
what do you do for a living
what happened yesterday
my name is sam
my name is sara and i am a student
my name is adam and i live in tunis
my favourite colour is blue
my favourite food is couscous
my favourite season is spring
my favourite sport is football
my favourite animal is the cat
my favourite subject is science
where are you from
where are you going
where do you live
where do you work
where is the station
where is the bathroom
where is my phone
where is the nearest bank
who are you
who is your best friend
who is that man
who wrote this book
why are you sad
why are you late
why is the sky blue
when is your birthday
when does the train leave
when does the shop open
i live in sfax
i live in a small house
i live with my family
i live near the sea
can you help me please
can you help me with my homework
can you help me find my keys
can you speak english
can you repeat that please
can you say that again
can i help you
can i have some water please
can i ask you a question
could you open the window please
please help me with my homework
please sit down
please come in
please be quiet
please wait a moment
please close the door
excuse me where is the station
excuse me can you help me
yes of course
yes i can
yes please
no thank you
no problem
no i do not
of course i can help you
i do not know
i do not understand
i do not like coffee
i do not have time
i do not think so
i think so
i think you are right
i think it is a good idea
i think therefore i am
i hope you are well
i hope to see you soon
i hope it does not rain
i know the answer
i know what you mean
i understand now
i agree with you
i believe in you
i miss you
i miss my family
i love you
i love you too
happy birthday to you
happy new year
congratulations on your new job
welcome to tunisia
welcome to our home
welcome back my friend
sorry i am late
sorry about that
sorry i did not hear you
do you speak english
do you speak french
do you like coffee
do you like music
do you have a pen
do you have any brothers or sisters
do you know the answer
do you understand
did you sleep well
did you eat breakfast
are you ready
are you ok
are you hungry
are you coming with us
is it far from here
is it raining
is this your book
it is nice to see you
it is very kind of you
it is time to sleep
it is time to eat
it is time to go home
it is time to wake up
it is never too late to learn
it is a beautiful day
it is a good idea
it is getting late
it is very cold outside
it is very hot today
it is raining today
it is sunny today
it is snowing in the mountains
it is windy today
it is cloudy this morning
it does not matter
let me help you
let me think about it
let us go to the beach
let us go home
let us start the lesson
let us eat together
`));

      /* ---------- home, family, daily life ---------- */
      add(...lines(`
i love my family
i love my mother
i love my father
i love my cat
i love my dog
i love my country
i love to read books
i love to play football
i love to write code
i love to travel
i love to cook for my family
i like to learn new things
i like to watch movies
i like to listen to music
i like to walk in the park
i like to drink tea in the evening
i want to learn english
i want to travel around the world
i want to drink some water
i want to be a doctor
i want to buy a new phone
i want to go home
i need to sleep
i need some help
i need a glass of water
i need to go to the bank
i need to finish my homework
i have a cat and a dog
i have two brothers and one sister
i have a big family
i have a lot of work today
i have a question
i have an idea
i have a headache
i drink coffee every morning
i drink tea in the evening
i eat breakfast every morning
i go to work every day
i go to school by bus
i go to the market on sunday
i go to bed at ten
i wake up at seven
i wake up early every morning
i brush my teeth every morning
i brush my teeth before bed
i take a shower in the morning
i get dressed and go to work
i come home in the evening
i cook dinner for my family
i do my homework after school
i read a book before bed
i write code every day
i work in an office
i work from home
i study at the university
i walk to school every day
she drinks coffee in the morning
she goes to the gym every day
she works in a bank
she works at the hospital
she is my sister
she is a good teacher
she is reading a book
she has a beautiful voice
she has long black hair
she lives in paris
he reads a book at night
he plays football with his friends
he works in a factory
he is my brother
he is a good man
he is watching television
he has a new car
he has a big dog
he lives in london
we eat dinner at night
we eat lunch at noon
we eat dinner together every evening
we are going to the beach
we are going to the cinema tonight
we are good friends
we are learning together
we are very happy today
we have a big garden
we have a lot of time
we live in a big city
we go to the beach in summer
they play football in the park
they are my friends
they are playing in the garden
they live in a small village
they have three children
they go to school by bus
the children are playing in the park
the children go to school every morning
the children are sleeping
the children love to play outside
the children are learning to read
the baby is sleeping
the baby is crying
the baby drinks milk
the boy plays with the ball
the boy is reading a book
the girl is singing a song
the girl plays the piano
the man is driving a car
the man is reading the newspaper
the woman is cooking dinner
the woman works in the city
the old man walks slowly
the old woman tells a story
the family eats dinner together
the family lives in a big house
the family goes on holiday in summer
my father works in a factory
my father drives a taxi
my father is a doctor
my mother cooks dinner every evening
my mother is a teacher
my mother makes the best couscous
my brother plays video games
my brother is older than me
my sister reads a lot of books
my sister is younger than me
my grandmother tells the best stories
my grandmother bakes bread every morning
my grandfather works in the garden
my grandfather is very wise
my uncle lives in france
my aunt has three children
my cousin is my best friend
my parents are very kind
my friend is very funny
my friend lives next door
my neighbour has a big dog
my cat is very cute
my cat sleeps all day
my dog is very friendly
my dog loves to run
my house is small but clean
my room is on the second floor
my car is in the garage
my phone is on the table
my bag is very heavy
the father of my father is my grandfather
the mother of my mother is my grandmother
the brother of my father is my uncle
the sister of my mother is my aunt
the son of my uncle is my cousin
the house is big
the house has a red door
the house has two floors and a garden
the room is clean
the kitchen is clean
the kitchen smells of fresh bread
the bedroom is upstairs
the bathroom is next to the bedroom
the garden is full of flowers
the door is open
the door is closed
the window is open
the light is on
the table is made of wood
the chair is broken
the bed is very comfortable
the book is on the table
the keys are on the table
the phone is on the desk
the cup is on the shelf
the clothes are in the wardrobe
the food is delicious
the food is ready
the coffee is hot
the tea is too hot
the water is cold
the soup is too salty
the bread is fresh
the cake is very sweet
the milk is in the fridge
the car is red
the car is fast
the car is in the garage
the bus is late
the bus stops near my house
the train is late again
the train leaves at eight
the train arrives at noon
the plane lands at the airport
the plane flies above the clouds
the ship sails across the sea
the boat is on the river
the city is very big
the city is full of people
the city never sleeps
the village is quiet
the street is empty at night
the street is very busy
the market is open every day
the shop is closed on sunday
the shop opens at nine
the school is near my house
the school is closed in summer
the hospital is far from here
the park is full of trees
the beach is beautiful in summer
the restaurant is full tonight
the museum is closed on monday
the library is very quiet
the road is long
the bridge is very old
the mountain is very high
the river is very long
the forest is dark at night
the desert is hot and dry
the island is small and green
breakfast is the first meal of the day
lunch is at noon
dinner is ready
dinner is at eight
`));

      /* ---------- animals: the classic examples ---------- */
      add(...lines(`
the cat sat on the mat
the dog sat on the rug
the cat sat on the sofa
the dog sat on the floor
the cat sat on the mat and slept
the cat sat on the chair
the cat chased the mouse
the dog chased the ball
the dog chased the cat
the cat is sleeping on the sofa
the dog is sleeping on the floor
the dog is playing in the park
the cat drinks milk
the dog eats meat
the bird sings in the tree
the bird flies in the sky
the bird builds a nest
the fish swims in the water
the horse runs in the field
the cow eats grass
the cow gives us milk
the sheep gives us wool
the bee makes honey
the hen lays eggs
the spider makes a web
the dog is a loyal friend
the dog wags its tail
the cat climbs the tree
the cat plays with a ball of wool
the mouse runs away from the cat
the lion is the king of the jungle
the elephant never forgets
the monkey climbs the tree
the fox is very clever
the quick brown fox jumps over the lazy dog
the early bird catches the worm
birds of a feather flock together
`));

      /* ---------- nature, weather, time ---------- */
      add(...lines(`
the weather is nice today
the weather is cold today
the weather is hot in summer
the weather is bad today
the weather is warm in spring
the sun rises in the morning
the sun rises in the east
the sun sets in the evening
the sun sets in the west
the sun is shining
the sun is a star
the sun gives us light and heat
the moon shines at night
the moon is full tonight
the moon goes around the earth
the stars shine at night
the stars are far away
the sky is blue
the sky is clear tonight
the sky is full of stars
the sky is grey today
the grass is green
the sea is blue and calm
the sea is deep
the sea is very cold in winter
the wind is strong today
the wind blows from the north
the rain falls from the clouds
the rain stopped an hour ago
the snow is white and cold
the snow covers the mountains
the clouds are grey
the trees are green in spring
the leaves fall in autumn
the leaves turn yellow in autumn
the flowers bloom in spring
the flowers are beautiful
the flowers need water and sunlight
the river flows to the sea
the earth goes around the sun
the earth is round
the earth is our home
the earth has one moon
the ocean is very deep
the world is big and beautiful
the world is full of wonders
the night is dark and quiet
the night is long in winter
the day is long in summer
the morning is cold
the evening is calm
in winter it is cold
in summer it is hot
in spring the flowers bloom
in autumn the leaves fall
in the morning i drink coffee
in the evening i read a book
in the afternoon we play football
at night the stars shine
at noon we eat lunch
water is the source of life
water freezes at zero degrees
water boils at one hundred degrees
ice is frozen water
plants need water and sunlight to grow
trees give us oxygen
fire is hot and ice is cold
time flies when you are having fun
time is money
time heals all wounds
today is a beautiful day
today is my birthday
today is the first day of school
tomorrow is another day
tomorrow will be better
yesterday was a long day
yesterday was very cold
there are seven days in a week
there are twelve months in a year
there are four seasons in a year
there are sixty minutes in an hour
there are sixty seconds in a minute
there are twenty four hours in a day
there are many stars in the sky
there is a cat on the roof
there is a book on the table
there is a park near my house
there is no place like home
there is always hope
a week has seven days
a year has twelve months
a year has four seasons
a day has twenty four hours
an hour has sixty minutes
a triangle has three sides
a square has four sides
a circle is round
a spider has eight legs
a bird has two wings
a car has four wheels
a bicycle has two wheels
a hand has five fingers
`));

      /* ---------- school, work, hobbies, travel ---------- */
      add(...lines(`
the teacher explains the lesson
the teacher writes on the board
the teacher asks a question
the teacher is very patient
the students listen to the teacher
the students write in their notebooks
the students are reading a story
the students answer the question
the students study for the exam
the student opens the book
the student raises his hand
the lesson starts at eight
the lesson is very interesting
the exam is next week
the exam was very hard
the homework is easy
the book is very interesting
the story has a happy ending
the story begins in a small village
the answer is correct
the answer is wrong
the question is very hard
the question is easy
the doctor works in the hospital
the doctor helps sick people
the chef cooks in the kitchen
the farmer works in the field
the driver stops the bus
the pilot flies the plane
the artist paints a picture
the singer sings a beautiful song
the writer writes a new book
the police officer helps people
the manager is in a meeting
the meeting starts at ten
the meeting is over
the office is closed today
the work is hard but interesting
the job is very important
the team works together
the team won the match
the match starts at eight
the game is over
the players are on the field
the ball is in the goal
football is the most popular sport in the world
music is the language of the soul
music makes me happy
reading is my favourite hobby
reading opens the mind
learning is fun
learning a language takes time
practice makes perfect
knowledge is power
honesty is the best policy
health is better than wealth
laughter is the best medicine
love is all you need
life is beautiful
life is short
life is a journey
friendship is a treasure
a friend in need is a friend indeed
a picture is worth a thousand words
a journey of a thousand miles begins with a single step
an apple a day keeps the doctor away
actions speak louder than words
all roads lead to rome
all that glitters is not gold
better late than never
better safe than sorry
easy come easy go
every cloud has a silver lining
no pain no gain
home sweet home
rome was not built in a day
two heads are better than one
the pen is mightier than the sword
where there is a will there is a way
when in rome do as the romans do
to be or not to be that is the question
roses are red and violets are blue
twinkle twinkle little star how i wonder what you are
once upon a time there was a king
once upon a time there was a little girl
once upon a time there was a small village
once upon a time there lived a wise old man
once upon a time in a land far away
the king lived in a big castle
the king had three sons
the queen was kind and wise
the princess lived in a tower
the prince rode a white horse
the dragon lived in a dark cave
the dragon guarded the treasure
the knight fought the dragon
the old wizard had a long white beard
they lived happily ever after
and they lived happily ever after
the end of the story is happy
i travel by train
i travel to work by bus
i want to visit paris
we visited the museum yesterday
we stayed in a small hotel
we took the train to the city
we walked along the beach
the hotel is near the beach
the ticket is very expensive
the ticket costs ten dinars
the trip was long but fun
the journey takes two hours
the airport is far from the city
the station is next to the bank
the bank is next to the post office
turn left at the corner
turn right at the traffic light
go straight ahead
`));

      /* ---------- how language models work (the theme of this site) ---------- */
      add(...lines(`
the model reads the sentence
the model reads one word at a time
the model predicts the next word
the model predicts the next token
the model predicts one word at a time
the model learns from data
the model learns from examples
the model learns patterns from text
the model was trained on a lot of text
the model has many layers
the model has millions of parameters
the model turns words into numbers
the model turns each word into a vector
the model gives every word a score
the model picks the most likely word
the model picks the next word
the model writes one word at a time
the model does not think like a human
the model makes mistakes sometimes
the model is only as good as its data
the model is a neural network
the model is called a transformer
the model pays attention to important words
the network has many layers
the network learns from data
the network learns its weights during training
the network is made of simple units
the network turns numbers into predictions
the next word depends on the words before it
the next word is chosen from the vocabulary
the next token is chosen at random
the attention mechanism compares every word with every other word
the attention score shows how much one word listens to another
the attention weights add up to one
the transformer is a kind of neural network
the transformer uses attention to read the sentence
the transformer was invented in two thousand seventeen
the tokenizer splits the text into tokens
the vocabulary is the list of all the words the model knows
the embedding is a list of numbers
the embedding stores the meaning of a word
the temperature controls how random the output is
the output of the model is a probability for each word
the input of the model is a list of tokens
the weights are learned during training
the training data is very large
the training takes a long time
the computer runs the program
the computer is very fast
the computer cannot think
the computer follows instructions
the program has a bug
the program runs on the server
the program is written in python
the code is easy to read
the code has a bug
the code runs in the browser
the server is running
the server is down
the internet connects the world
the internet is full of information
the data is stored in the cloud
the robot can walk and talk
the future is full of possibilities
attention helps the model focus on important words
attention is all you need
attention lets each word look at the other words
each word looks at the other words
each word becomes a vector
each word gets a score
each token gets a number
each layer makes the meaning clearer
self attention compares every word with every other word
transformers use self attention
transformers changed the world of artificial intelligence
transformers read all the words at once
language models learn patterns from text
language models predict the next word
language models are trained on a lot of text
language models can write stories and answer questions
language is a tool for thinking
large language models have billions of parameters
artificial intelligence is changing the world
artificial intelligence is a branch of computer science
artificial intelligence helps doctors and teachers
machine learning is a part of artificial intelligence
machine learning is the science of learning from data
deep learning uses neural networks with many layers
neural networks learn from data
neural networks are inspired by the brain
a neural network is made of layers
a language model predicts the next word
a token is a piece of a word
a vector is a list of numbers
a matrix is a table of numbers
a transformer is a kind of neural network
a computer can only work with numbers
a good model needs a lot of data
a higher temperature makes the output more random
a lower temperature makes the output more predictable
words with similar meanings have similar vectors
words are turned into numbers
numbers are easier for a computer than words
python is a programming language
python is easy to learn
javascript runs in the browser
programming is fun
programming is like solving a puzzle
data is the new oil
training a model takes a lot of data
training a model takes time and energy
i write code every day
i love to write code
i am learning machine learning
i am learning how language models work
i want to understand how artificial intelligence works
i asked the model a question
the answer of the model was correct
`));

      /* ---------- a little French ---------- */
      add(...lines(`
bonjour comment ça va
bonjour comment allez vous
bonjour comment vas tu
bonjour tout le monde
bonjour mon ami
bonjour et bienvenue
bonsoir tout le monde
bonsoir mon ami
salut ça va
salut comment ça va
ça va bien merci
ça va très bien merci
ça va bien et toi
je vais bien merci
je vais très bien merci
je vais à l'école
je vais au travail
je vais au marché
merci beaucoup
merci beaucoup mon ami
merci pour ton aide
de rien mon ami
s'il vous plaît
au revoir et à bientôt
au revoir mon ami
à bientôt mon ami
à demain
bonne nuit
bonne nuit et fais de beaux rêves
bonne journée
bonne journée à tous
bonne chance mon ami
bon appétit
bon voyage
comment tu t'appelles
comment vous appelez vous
je m'appelle sam
je m'appelle sara
je suis content de te voir
je suis fatigué
je suis étudiant
je suis tunisien
je suis de tunisie
j'habite à sfax
j'habite en tunisie
j'aime ma famille
j'aime le café
j'aime lire des livres
j'aime la musique
j'aime jouer au football
je t'aime
je ne sais pas
je ne comprends pas
je parle français
je parle un peu français
je bois du café le matin
je bois du thé le soir
je mange une pomme
je mange du pain
je mange du couscous
je lis un livre
je veux apprendre le français
je veux un café s'il vous plaît
le chat dort sur le canapé
le chat boit du lait
le chat est sur la table
le chien joue dans le jardin
le chien mange de la viande
le chien court dans le parc
le soleil brille
le soleil se lève le matin
le ciel est bleu
le ciel est gris
la mer est bleue
la mer est calme
la lune brille la nuit
la maison est grande
la maison est petite
la vie est belle
la porte est ouverte
la fille lit un livre
le garçon joue au football
le livre est sur la table
le café est chaud
le modèle prédit le mot suivant
il fait beau aujourd'hui
il fait froid aujourd'hui
il fait chaud en été
il pleut aujourd'hui
il est tard
il est temps de dormir
elle est très gentille
nous sommes amis
nous allons à la plage
où est la gare
où habites tu
quelle heure est il
oui bien sûr
non merci
c'est la vie
c'est très bien
c'est une bonne idée
`));

      /* ---------- fact tables, expanded into sentences ---------- */

      // animal, sound it makes, what it eats, where it lives, what it is like
      const ANIMALS = [
        ['dog', 'barks', 'meat', 'in a house', 'loyal'], ['cat', 'meows', 'fish', 'in a house', 'curious'],
        ['cow', 'moos', 'grass', 'on a farm', 'gentle'], ['horse', 'neighs', 'hay', 'in a stable', 'fast'],
        ['sheep', 'bleats', 'grass', 'in a field', 'woolly'], ['goat', 'bleats', 'leaves', 'in the mountains', 'stubborn'],
        ['pig', 'oinks', 'corn', 'on a farm', 'pink'], ['donkey', 'brays', 'hay', 'in a stable', 'patient'],
        ['chicken', 'clucks', 'grain', 'on a farm', 'small'], ['rooster', 'crows', 'grain', 'on a farm', 'loud'],
        ['duck', 'quacks', 'bread', 'near a pond', 'noisy'], ['frog', 'croaks', 'flies', 'near a pond', 'green'],
        ['lion', 'roars', 'meat', 'in the savanna', 'brave'], ['tiger', 'growls', 'meat', 'in the jungle', 'striped'],
        ['elephant', 'trumpets', 'leaves', 'in the savanna', 'huge'], ['giraffe', 'hums', 'leaves', 'in the savanna', 'tall'],
        ['zebra', 'barks', 'grass', 'in the savanna', 'striped'], ['monkey', 'chatters', 'bananas', 'in the jungle', 'clever'],
        ['wolf', 'howls', 'meat', 'in the forest', 'wild'], ['bear', 'growls', 'honey', 'in the forest', 'strong'],
        ['fox', 'yelps', 'rabbits', 'in the forest', 'sly'], ['deer', 'grunts', 'grass', 'in the forest', 'shy'],
        ['rabbit', 'squeaks', 'carrots', 'in a burrow', 'soft'], ['mouse', 'squeaks', 'cheese', 'in a hole', 'tiny'],
        ['owl', 'hoots', 'mice', 'in a tree', 'wise'], ['eagle', 'screams', 'fish', 'in the mountains', 'proud'],
        ['parrot', 'talks', 'seeds', 'in the jungle', 'colourful'], ['pigeon', 'coos', 'seeds', 'in the city', 'grey'],
        ['bee', 'buzzes', 'nectar', 'in a hive', 'busy'], ['snake', 'hisses', 'mice', 'in the desert', 'long'],
        ['camel', 'grunts', 'plants', 'in the desert', 'patient'], ['dolphin', 'whistles', 'fish', 'in the ocean', 'playful'],
        ['whale', 'sings', 'plankton', 'in the ocean', 'enormous'], ['shark', 'hunts', 'fish', 'in the ocean', 'dangerous'],
        ['penguin', 'squawks', 'fish', 'on the ice', 'funny'], ['turtle', 'hisses', 'plants', 'near the sea', 'slow'],
        ['kangaroo', 'jumps', 'grass', 'in australia', 'strong'], ['panda', 'squeaks', 'bamboo', 'in china', 'cute'],
        ['crocodile', 'growls', 'fish', 'in the river', 'dangerous'], ['butterfly', 'flutters', 'nectar', 'in the garden', 'beautiful']
      ];
      ANIMALS.forEach(([a, sound, food, home, trait], i) => {
        add(`the ${a} ${sound}`, `the ${a} eats ${food}`, `the ${a} lives ${home}`, `the ${a} is ${an(trait)} animal`, `the ${a} is very ${trait}`,
          [`i saw ${an(a)} at the zoo`, `my favourite animal is the ${a}`, `the children love the ${a}`, `${an(a)} is an animal`, `have you ever seen ${an(a)}`][i % 5]);
      });

      // country, capital, language, continent
      const COUNTRIES = [
        ['tunisia', 'tunis', 'arabic', 'africa'], ['france', 'paris', 'french', 'europe'], ['italy', 'rome', 'italian', 'europe'],
        ['spain', 'madrid', 'spanish', 'europe'], ['germany', 'berlin', 'german', 'europe'], ['england', 'london', 'english', 'europe'],
        ['portugal', 'lisbon', 'portuguese', 'europe'], ['greece', 'athens', 'greek', 'europe'], ['russia', 'moscow', 'russian', 'europe'],
        ['sweden', 'stockholm', 'swedish', 'europe'], ['norway', 'oslo', 'norwegian', 'europe'], ['poland', 'warsaw', 'polish', 'europe'],
        ['austria', 'vienna', 'german', 'europe'], ['ireland', 'dublin', 'english', 'europe'], ['belgium', 'brussels', 'french', 'europe'],
        ['egypt', 'cairo', 'arabic', 'africa'], ['morocco', 'rabat', 'arabic', 'africa'], ['algeria', 'algiers', 'arabic', 'africa'],
        ['libya', 'tripoli', 'arabic', 'africa'], ['senegal', 'dakar', 'french', 'africa'], ['kenya', 'nairobi', 'swahili', 'africa'],
        ['nigeria', 'abuja', 'english', 'africa'], ['japan', 'tokyo', 'japanese', 'asia'], ['china', 'beijing', 'chinese', 'asia'],
        ['india', 'delhi', 'hindi', 'asia'], ['korea', 'seoul', 'korean', 'asia'], ['turkey', 'ankara', 'turkish', 'asia'],
        ['lebanon', 'beirut', 'arabic', 'asia'], ['jordan', 'amman', 'arabic', 'asia'], ['iraq', 'baghdad', 'arabic', 'asia'],
        ['qatar', 'doha', 'arabic', 'asia'], ['thailand', 'bangkok', 'thai', 'asia'], ['vietnam', 'hanoi', 'vietnamese', 'asia'],
        ['canada', 'ottawa', 'english', 'america'], ['mexico', 'mexico city', 'spanish', 'america'], ['brazil', 'brasilia', 'portuguese', 'america'],
        ['argentina', 'buenos aires', 'spanish', 'america'], ['peru', 'lima', 'spanish', 'america'], ['cuba', 'havana', 'spanish', 'america'],
        ['australia', 'canberra', 'english', 'oceania']
      ];
      COUNTRIES.forEach(([c, cap, lang, cont], i) => {
        add(`${cap} is the capital of ${c}`, `the capital of ${c} is ${cap}`, `people in ${c} speak ${lang}`, `${c} is a country in ${cont}`,
          [`i want to visit ${cap}`, `my friend lives in ${cap}`, `we travelled to ${c} last summer`, `she comes from ${c}`, `he was born in ${cap}`, `they speak ${lang} in ${cap}`][i % 6],
          [`${cap} is a beautiful city`, `${c} is a beautiful country`, `i have never been to ${c}`, `the food in ${c} is delicious`][i % 4]);
      });
      add(...lines(`
tunisia is a small country in north africa
tunisia is famous for its olive oil
tunisia has beautiful beaches
sfax is a city in tunisia
sfax is famous for its olive trees
tunis is the largest city in tunisia
paris is the city of light
paris is famous for the eiffel tower
rome is a very old city
london is a big city
egypt is famous for the pyramids
china is the largest country in asia
the sahara is the largest desert in the world
the nile is the longest river in africa
the pacific is the largest ocean in the world
africa is a continent
europe is a continent
asia is the largest continent
english is spoken all over the world
arabic is written from right to left
french is a beautiful language
`));

      // sequences: days, months, seasons, numbers
      const seq = (xs, unit, ord) => xs.forEach((x, i) => {
        const nx = xs[(i + 1) % xs.length];
        add(`after ${x} comes ${nx}`, `${x} comes before ${nx}`);
        if (ord && ord[i]) add(`${x} is the ${ord[i]} ${unit}`);
      });
      const ORD = ['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth', 'eleventh', 'twelfth'];
      seq(['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'], 'day of the week', ORD);
      seq(['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december'], 'month of the year', ORD);
      seq(['spring', 'summer', 'autumn', 'winter'], 'season', null);
      const NUM = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty'];
      for (let a = 1; a < 20; a++) add(`after ${NUM[a]} comes ${NUM[a + 1]}`);
      for (let a = 1; a <= 10; a++) for (let b = 1; b <= 10 && a + b <= 20; b++) add(`${NUM[a]} plus ${NUM[b]} is ${NUM[a + b]}`);
      for (let a = 2; a <= 10; a++) for (let b = 1; b < a && b <= 5; b++) add(`${NUM[a]} minus ${NUM[b]} is ${NUM[a - b]}`);
      for (let a = 2; a <= 5; a++) for (let b = 2; b <= 5 && a * b <= 20; b++) add(`${NUM[a]} times ${NUM[b]} is ${NUM[a * b]}`);
      add(...lines(`
one two three four five six seven eight nine ten
ten nine eight seven six five four three two one
i can count from one to ten
ten plus ten is twenty
twenty is two times ten
one hundred is ten times ten
one thousand is ten times one hundred
one million is a very big number
saturday and sunday are the weekend
friday is my favourite day
on monday i go back to work
on sunday we visit our grandparents
ramadan is the month of fasting
my birthday is in july
school starts in september
the new year begins in january
winter is the coldest season
summer is the hottest season
spring is the season of flowers
autumn is the season of falling leaves
`));

      // thing, colour
      const COLOURS = [
        ['sky', 'blue'], ['sea', 'blue'], ['grass', 'green'], ['leaf', 'green'], ['sun', 'yellow'], ['banana', 'yellow'], ['lemon', 'yellow'],
        ['snow', 'white'], ['milk', 'white'], ['cloud', 'white'], ['coal', 'black'], ['night', 'black'], ['blood', 'red'], ['tomato', 'red'],
        ['rose', 'red'], ['strawberry', 'red'], ['carrot', 'orange'], ['pumpkin', 'orange'], ['chocolate', 'brown'], ['wood', 'brown'],
        ['elephant', 'grey'], ['flamingo', 'pink'], ['eggplant', 'purple'], ['gold', 'yellow'], ['emerald', 'green']
      ];
      COLOURS.forEach(([t, c], i) => add(`the ${t} is ${c}`, [`the colour of the ${t} is ${c}`, `${an(t)} is ${c}`, `${t} is ${c}`][i % 3]));
      add(...lines(`
red and yellow make orange
blue and yellow make green
red and blue make purple
black and white make grey
red and white make pink
the rainbow has seven colours
the flag of tunisia is red and white
the traffic light is red
green means go and red means stop
my favourite colour is green
her favourite colour is purple
his favourite colour is red
`));

      // opposites
      const OPP = [
        ['hot', 'cold'], ['big', 'small'], ['fast', 'slow'], ['happy', 'sad'], ['good', 'bad'], ['day', 'night'], ['up', 'down'], ['open', 'closed'],
        ['light', 'dark'], ['old', 'young'], ['rich', 'poor'], ['strong', 'weak'], ['early', 'late'], ['easy', 'hard'], ['full', 'empty'],
        ['high', 'low'], ['long', 'short'], ['wet', 'dry'], ['clean', 'dirty'], ['love', 'hate'], ['left', 'right'], ['true', 'false'],
        ['yes', 'no'], ['first', 'last'], ['loud', 'quiet'], ['thick', 'thin'], ['near', 'far'], ['inside', 'outside'], ['before', 'after'],
        ['always', 'never'], ['black', 'white'], ['north', 'south'], ['east', 'west'], ['top', 'bottom'], ['sweet', 'bitter'], ['cheap', 'expensive'],
        ['safe', 'dangerous'], ['asleep', 'awake'], ['question', 'answer'], ['friend', 'enemy'], ['war', 'peace'], ['beginning', 'end'],
        ['push', 'pull'], ['give', 'take'], ['buy', 'sell'], ['win', 'lose'], ['laugh', 'cry'], ['start', 'stop'], ['heavy', 'light'], ['new', 'old']
      ];
      OPP.forEach(([a, b]) => add(`the opposite of ${a} is ${b}`, `the opposite of ${b} is ${a}`));

      // job, what they do, where they work
      const JOBS = [
        ['teacher', 'teaches the children', 'at the school'], ['doctor', 'treats sick people', 'in the hospital'], ['nurse', 'cares for the patients', 'in the hospital'],
        ['chef', 'cooks delicious food', 'in the restaurant'], ['baker', 'bakes bread', 'in the bakery'], ['farmer', 'grows vegetables', 'on the farm'],
        ['pilot', 'flies the plane', 'at the airport'], ['driver', 'drives the bus', 'in the city'], ['firefighter', 'puts out fires', 'at the fire station'],
        ['writer', 'writes books', 'at home'], ['painter', 'paints pictures', 'in the studio'], ['singer', 'sings songs', 'on the stage'],
        ['actor', 'plays in movies', 'in the theatre'], ['engineer', 'builds bridges', 'in the office'], ['programmer', 'writes code', 'in the office'],
        ['scientist', 'does experiments', 'in the laboratory'], ['fisherman', 'catches fish', 'at the sea'], ['mechanic', 'repairs cars', 'in the garage'],
        ['dentist', 'fixes teeth', 'in the clinic'], ['lawyer', 'defends people', 'in the court'], ['judge', 'makes decisions', 'in the court'],
        ['soldier', 'defends the country', 'in the army'], ['sailor', 'sails the ship', 'at the port'], ['tailor', 'sews clothes', 'in the workshop'],
        ['carpenter', 'makes furniture', 'in the workshop'], ['plumber', 'fixes pipes', 'in the houses'], ['librarian', 'lends books', 'in the library'],
        ['waiter', 'serves the food', 'in the restaurant'], ['pharmacist', 'sells medicine', 'in the pharmacy'], ['journalist', 'writes the news', 'at the newspaper'],
        ['photographer', 'takes photos', 'in the studio'], ['musician', 'plays music', 'on the stage'], ['gardener', 'waters the plants', 'in the garden'],
        ['barber', 'cuts hair', 'in the barbershop'], ['vet', 'treats sick animals', 'in the clinic'], ['astronaut', 'travels to space', 'at the space station'],
        ['architect', 'designs buildings', 'in the office'], ['postman', 'delivers letters', 'at the post office'], ['butcher', 'sells meat', 'in the market'],
        ['shepherd', 'watches the sheep', 'in the mountains'], ['student', 'studies hard', 'at the university'], ['guard', 'protects the building', 'at the gate']
      ];
      const KIN = ['father', 'mother', 'brother', 'sister', 'uncle', 'aunt', 'friend', 'cousin', 'neighbour', 'grandfather'];
      JOBS.forEach(([j, does, place], i) => add(`the ${j} ${does}`, `${an(j)} ${does}`, `the ${j} works ${place}`,
        [`my ${KIN[i % 10]} is ${an(j)}`, `i want to be ${an(j)}`, `she wants to be ${an(j)}`, `he is a very good ${j}`][i % 4]));

      // food, what kind of thing it is, what it is like
      const FOODS = [
        ['apple', 'fruit', 'sweet'], ['banana', 'fruit', 'sweet'], ['orange', 'fruit', 'juicy'], ['lemon', 'fruit', 'sour'], ['grape', 'fruit', 'sweet'],
        ['strawberry', 'fruit', 'sweet'], ['watermelon', 'fruit', 'juicy'], ['mango', 'fruit', 'sweet'], ['peach', 'fruit', 'soft'], ['pear', 'fruit', 'juicy'],
        ['cherry', 'fruit', 'small'], ['date', 'fruit', 'sweet'], ['fig', 'fruit', 'sweet'], ['olive', 'fruit', 'salty'], ['pomegranate', 'fruit', 'red'],
        ['tomato', 'vegetable', 'red'], ['potato', 'vegetable', 'cheap'], ['carrot', 'vegetable', 'crunchy'], ['onion', 'vegetable', 'strong'],
        ['cucumber', 'vegetable', 'fresh'], ['pepper', 'vegetable', 'spicy'], ['garlic', 'vegetable', 'strong'], ['spinach', 'vegetable', 'healthy'],
        ['lettuce', 'vegetable', 'fresh'], ['pumpkin', 'vegetable', 'big']
      ];
      FOODS.forEach(([x, kind, adj], i) => add(`${an(x)} is a ${kind}`, `the ${x} is ${adj}`,
        [`i ate ${an(x)} this morning`, `she bought ${an(x)} at the market`, `the ${x} is my favourite ${kind}`, `he does not like the taste of ${x}`, `we need one more ${x} for the salad`][i % 5]));
      // dish or drink, what it is like, when or how people have it
      const DISHES = [
        ['bread', 'fresh', 'for breakfast'], ['rice', 'white', 'for lunch'], ['pasta', 'delicious', 'for dinner'], ['couscous', 'delicious', 'on friday'],
        ['pizza', 'hot', 'for dinner'], ['soup', 'warm', 'in winter'], ['salad', 'fresh', 'for lunch'], ['cheese', 'salty', 'with bread'],
        ['chicken', 'tasty', 'with rice'], ['fish', 'fresh', 'with lemon'], ['meat', 'tender', 'with potatoes'], ['honey', 'sweet', 'with bread'],
        ['chocolate', 'sweet', 'after dinner'], ['cake', 'sweet', 'on my birthday'], ['ice cream', 'cold', 'in summer'], ['coffee', 'bitter', 'in the morning'],
        ['tea', 'hot', 'in the evening'], ['milk', 'white', 'before bed'], ['juice', 'sweet', 'for breakfast'], ['water', 'fresh', 'all day'],
        ['yogurt', 'healthy', 'for breakfast'], ['butter', 'soft', 'with bread'], ['jam', 'sweet', 'with bread'], ['sugar', 'sweet', 'in my tea'],
        ['salt', 'salty', 'in the soup'], ['olive oil', 'healthy', 'with bread']
      ];
      DISHES.forEach(([x, adj, when], i) => add(`${x} is ${adj}`, `i like ${x} ${when}`,
        [`we had ${x} ${when}`, `she loves ${x}`, `he never eats ${x}`, `would you like some ${x}`, `there is no more ${x} in the kitchen`, `my mother makes the best ${x}`][i % 6]));

      // tool, what we do with it
      const TOOLS = [
        ['knife', 'cut the bread'], ['pen', 'write a letter'], ['pencil', 'draw a picture'], ['spoon', 'eat the soup'], ['fork', 'eat the pasta'],
        ['key', 'open the door'], ['phone', 'call our friends'], ['computer', 'write code'], ['camera', 'take photos'], ['clock', 'tell the time'],
        ['umbrella', 'stay dry in the rain'], ['broom', 'sweep the floor'], ['hammer', 'hit the nails'], ['toothbrush', 'brush our teeth'],
        ['towel', 'dry our hands'], ['oven', 'bake a cake'], ['fridge', 'keep the food cold'], ['lamp', 'light the room'], ['map', 'find the way'],
        ['telescope', 'look at the stars'], ['microscope', 'see very small things'], ['ladder', 'climb to the roof'], ['needle', 'sew the clothes'],
        ['ruler', 'measure the line'], ['calculator', 'do the maths'], ['backpack', 'carry our books'], ['bucket', 'carry the water'],
        ['comb', 'brush our hair'], ['cup', 'drink the tea'], ['pan', 'fry the eggs'], ['shovel', 'dig a hole'], ['rope', 'pull the boat'],
        ['dictionary', 'find the meaning of a word'], ['thermometer', 'measure the temperature'], ['compass', 'find the north'], ['helmet', 'protect the head']
      ];
      TOOLS.forEach(([t, use], i) => add(`we use ${an(t)} to ${use}`, [`you need ${an(t)} to ${use}`, `i need ${an(t)} to ${use}`, `she used ${an(t)} to ${use}`][i % 3]));
      add(...lines(`
we use soap to wash our hands
we use scissors to cut the paper
we use glasses to see better
we use money to buy things
we use words to share ideas
we see with our eyes
we hear with our ears
we smell with our nose
we taste with our tongue
we walk with our legs
we hold things with our hands
we think with our brain
we chew with our teeth
we breathe with our lungs
the heart pumps blood around the body
the brain controls the body
the eyes are the window of the soul
you can buy bread at the bakery
you can buy medicine at the pharmacy
you can buy vegetables at the market
you can borrow books from the library
you can watch a movie at the cinema
you can catch a train at the station
you can swim in the sea
you can see animals at the zoo
we sleep in the bedroom
we cook in the kitchen
we wash in the bathroom
we eat in the dining room
we keep our money in the bank
we pray in the mosque
we learn at school
planes land at the airport
ships arrive at the port
trains stop at the station
mercury is the closest planet to the sun
venus is the hottest planet
mars is the red planet
jupiter is the biggest planet
saturn has beautiful rings
the earth is the third planet from the sun
the sun is the centre of the solar system
the moon is smaller than the earth
light travels faster than sound
a year on earth has three hundred and sixty five days
`));

      // everyday actions: base form, he/she form, -ing form, the rest of the phrase
      const ACTS = [
        ['read', 'reads', 'reading', 'a book'], ['play', 'plays', 'playing', 'football'], ['cook', 'cooks', 'cooking', 'dinner'],
        ['watch', 'watches', 'watching', 'a movie'], ['listen', 'listens', 'listening', 'to music'], ['go', 'goes', 'going', 'for a walk'],
        ['ride', 'rides', 'riding', 'a bike'], ['learn', 'learns', 'learning', 'english'], ['write', 'writes', 'writing', 'a letter'],
        ['visit', 'visits', 'visiting', 'the museum'], ['drink', 'drinks', 'drinking', 'some tea'], ['take', 'takes', 'taking', 'a nap'],
        ['swim', 'swims', 'swimming', 'in the sea'], ['travel', 'travels', 'travelling', 'by train'], ['study', 'studies', 'studying', 'for the exam'],
        ['clean', 'cleans', 'cleaning', 'the house'], ['wash', 'washes', 'washing', 'the dishes'], ['buy', 'buys', 'buying', 'some bread'],
        ['drive', 'drives', 'driving', 'the car'], ['paint', 'paints', 'painting', 'a picture'], ['sing', 'sings', 'singing', 'a song'],
        ['dance', 'dances', 'dancing', 'at the party'], ['run', 'runs', 'running', 'in the park'], ['sleep', 'sleeps', 'sleeping', 'early'],
        ['work', 'works', 'working', 'in the garden'], ['call', 'calls', 'calling', 'a friend'], ['open', 'opens', 'opening', 'the window'],
        ['close', 'closes', 'closing', 'the door'], ['eat', 'eats', 'eating', 'an apple'], ['make', 'makes', 'making', 'a cake'],
        ['build', 'builds', 'building', 'a house'], ['fix', 'fixes', 'fixing', 'the bike'], ['draw', 'draws', 'drawing', 'a map'],
        ['plant', 'plants', 'planting', 'a tree'], ['climb', 'climbs', 'climbing', 'the mountain'], ['catch', 'catches', 'catching', 'the bus'],
        ['bake', 'bakes', 'baking', 'some bread'], ['feed', 'feeds', 'feeding', 'the cat'], ['water', 'waters', 'watering', 'the plants'],
        ['play', 'plays', 'playing', 'the guitar'], ['play', 'plays', 'playing', 'chess'], ['speak', 'speaks', 'speaking', 'three languages'],
        ['wear', 'wears', 'wearing', 'a red hat'], ['carry', 'carries', 'carrying', 'a heavy bag'], ['send', 'sends', 'sending', 'an email'],
        ['answer', 'answers', 'answering', 'the phone'], ['ask', 'asks', 'asking', 'a question'], ['tell', 'tells', 'telling', 'a story'],
        ['wait', 'waits', 'waiting', 'for the bus'], ['look', 'looks', 'looking', 'at the stars'], ['think', 'thinks', 'thinking', 'about the future'],
        ['dream', 'dreams', 'dreaming', 'of the sea'], ['help', 'helps', 'helping', 'the neighbours'], ['teach', 'teaches', 'teaching', 'the children'],
        ['count', 'counts', 'counting', 'the stars'], ['pick', 'picks', 'picking', 'some flowers'], ['cut', 'cuts', 'cutting', 'the bread'],
        ['pour', 'pours', 'pouring', 'the tea'], ['share', 'shares', 'sharing', 'the cake'], ['save', 'saves', 'saving', 'some money'],
        ['walk', 'walks', 'walking', 'to school'], ['sit', 'sits', 'sitting', 'under the tree'], ['stay', 'stays', 'staying', 'at home'],
        ['meet', 'meets', 'meeting', 'new people'], ['find', 'finds', 'finding', 'a solution'], ['solve', 'solves', 'solving', 'the problem'],
        ['practise', 'practises', 'practising', 'the piano'], ['enjoy', 'enjoys', 'enjoying', 'the sunshine'], ['remember', 'remembers', 'remembering', 'the old days'],
        ['explore', 'explores', 'exploring', 'the old city']
      ];
      const FRAMES = [
        (b, s, g, r) => `i like to ${b} ${r}`, (b, s, g, r) => `she likes to ${b} ${r}`, (b, s, g, r) => `he ${s} ${r} every day`,
        (b, s, g, r) => `we are ${g} ${r}`, (b, s, g, r) => `they want to ${b} ${r}`, (b, s, g, r) => `i am ${g} ${r}`,
        (b, s, g, r) => `my friend ${s} ${r}`, (b, s, g, r) => `do you want to ${b} ${r}`, (b, s, g, r) => `she is ${g} ${r} now`,
        (b, s, g, r) => `we ${b} ${r} on sunday`, (b, s, g, r) => `i need to ${b} ${r}`, (b, s, g, r) => `he is going to ${b} ${r}`,
        (b, s, g, r) => `let us ${b} ${r}`, (b, s, g, r) => `can you ${b} ${r}`, (b, s, g, r) => `they ${b} ${r} together`,
        (b, s, g, r) => `i will ${b} ${r} tomorrow`, (b, s, g, r) => `she ${s} ${r} every morning`, (b, s, g, r) => `we love to ${b} ${r}`,
        (b, s, g, r) => `my brother ${s} ${r}`, (b, s, g, r) => `i did not ${b} ${r} yesterday`
      ];
      ACTS.forEach((a, i) => {for (let j = 0; j < 6; j++) add(FRAMES[(i * 7 + j * 3) % FRAMES.length](...a));});

      // describing people and things
      const DESC = [
        ['the teacher', 'kind'], ['the student', 'clever'], ['the baby', 'happy'], ['the old man', 'wise'], ['the girl', 'brave'], ['the boy', 'tall'],
        ['the story', 'interesting'], ['the movie', 'funny'], ['the song', 'beautiful'], ['the lesson', 'easy'], ['the problem', 'difficult'],
        ['the idea', 'brilliant'], ['the room', 'quiet'], ['the street', 'noisy'], ['the bag', 'heavy'], ['the box', 'empty'], ['the glass', 'full'],
        ['the road', 'dangerous'], ['the night', 'calm'], ['the view', 'wonderful'], ['the price', 'high'], ['the answer', 'simple'],
        ['the garden', 'lovely'], ['the weather', 'terrible'], ['the water', 'clear'], ['the mountain', 'huge'], ['the city', 'modern'], ['the village', 'peaceful']
      ];
      DESC.forEach(([x, adj], i) => add(`${x} is ${adj}`, [`${x} is very ${adj} today`, `${x} is not ${adj}`, `i think ${x} is ${adj}`, `${x} was ${adj} yesterday`][i % 4]));

      /* ---------- thousands of short story sentences (corpus-stories.js, optional) ---------- */
      if (typeof STORY_TEXT === 'string') for (const s of STORY_TEXT.split('\n')) out.push(s);

      return out.join('\n');
    })();
