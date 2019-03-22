#setwd("C:/quants/books/R/CRC.Data.Science.in.R/PairsTrading/")
setwd("C:/quants/book/R/CRC.Data.Science.in.R/PairsTrading/")
library(XML)
xmlSourceFunctions("duncanSol.Rdb")
readData =
#
# A function to read the data and convert the Date column
# to an object of class Date.
# The date values are expected to be in a column named Date.
# We may want to relax this and allow the caller specify the
# column - by name or index.
function(fileName, dateFormat = c("%Y-%m-%d", "%Y/%m/%d"), ...)
{
data = read.csv(fileName, header = TRUE,
stringsAsFactors = FALSE, ...)
for(fmt in dateFormat) {
tmp = as.Date(data$Date, fmt)
if(all(!is.na(tmp))) {
data$Date = tmp
break
}
}
data[ order(data$Date), ]
}
dji = readData("DJIA.csv", na.strings = ".")
sp500 = readData("SP500.csv", na.strings = ".")

dates = intersect(dji$Date, sp500$Date)
both = data.frame(Date = as.Date(dates, origin = "1970-01-01"), DJI = dji$VALUE[ dji$Date %in% dates], SP = sp500$VALUE[ sp500$Date %in% dates])

both$ratio = both$DJI/both$SP
both = both[!is.na(both$ratio),]

both95 = both[ both$Date <= as.Date("1994-12-31"), ]


m = mean(both95$ratio)
sd = sd(both95$ratio)

both2010 = both[ both$Date <= as.Date("2010-1-1"), ]

readData =
  #
  # A function to read the data and convert the Date column 
  # to an object of class Date.  
  # The date values are expected to be in a column named Date.
  # We may want to relax this and allow the caller specify the
  # column - by name or index.
function(fileName, dateFormat = c("%Y-%m-%d", "%Y/%m/%d"), ...)
{
   data = read.csv(fileName, header = TRUE, 
                     stringsAsFactors = FALSE, ...)
   for(fmt in dateFormat) {
      tmp = as.Date(data$Date, fmt)
      if(all(!is.na(tmp))) {
         data$Date = tmp
         break
      }
   }

   data[ order(data$Date), ]
}
#################################

library("quantmod")


toload<-"VZ"
dfload <- getSymbols(toload,from="1970-01-01",to="2013-11-30",auto.assign=FALSE)
   dfload <- data.frame(Date = index(dfload), dfload, row.names=NULL)
   
   colnames(dfload) <- c("Date","Open", "High", "Low", "Close","Volume","Adj Close")
 write.csv(dfload, file = paste0("VERIZON", ".csv"),row.names = FALSE) 
 

toload<-"T"
dfload <- getSymbols(toload,from="1970-01-01",to="2013-11-30",auto.assign=FALSE)
   dfload <- data.frame(Date = index(dfload), dfload, row.names=NULL)
   
   colnames(dfload) <- c("Date","Open", "High", "Low", "Close","Volume","Adj Close")
 write.csv(dfload, file = paste0("ATT", ".csv"),row.names = FALSE) 
 
#################??תծ???ݻ?ȡ######################################
 toload<-"600185.ss"
 toload<-"600036.ss"
 dfload <- getSymbols(toload,from="2017-01-01",to="2018-10-25",auto.assign=FALSE)
 dfload <- data.frame(Date = index(dfload), dfload, row.names=NULL)
 colnames(dfload) <- c("Date","Open", "High", "Low", "Close","Volume","Adj Close")
 write.csv(dfload, file = paste0(toload, ".csv"),row.names = FALSE) 
 #rm(toload,dfload)
 #toload<-"110030.ss"
 #dfload <- getSymbols(toload,from="2017-01-01",to="2017-10-25",auto.assign=FALSE)
 #dfload <- data.frame(Date = index(dfload), dfload, row.names=NULL)
 #colnames(dfload) <- c("Date","Open", "High", "Low", "Close","Volume","Adj Close")
 #(dfload, file = paste0(toload, ".csv"),row.names = FALSE) 
####################????ȡ???ݳ?ʢ????
 library(WindR)
 w.start()
 toload<-"110030.ss"
 dfload<-w.wsd("110030.SH","open,high,low,close,volume","20140101","20170825")
 dfload_trans<-dfload$Data[c(1,2,3,4,5,6,5)]
 colnames(dfload_trans) <- c("Date","Open", "High", "Low", "Close","Volume","Adj Close")
 dfload_trans<-as.data.frame(dfload_trans)
 dfload_trans<-dfload_trans[which(dfload_trans$Volume>0),]
 write.csv(dfload_trans, file = paste0(toload, ".csv"),row.names = FALSE) 
 ##########################################
ss600036 <- readData("600036.CSV",na.strings = ".")         # ss110030 symbol
ss600185 = readData("600185.CSV",na.strings = ".") # ss600185 symbol


##########?????????ݱ?????ͬ????
dates = intersect(ss600036$Date, ss600185$Date)

ss600036<-ss600036[which(ss600036$Date %in% dates),]
ss600185<-ss600185[which(ss600185$Date %in% dates),]



combine2Stocks = 
function(a, b, stockNames = c(deparse(substitute(a)), 
                              deparse(substitute(b))))
{
  rr = range(intersect(a$Date, b$Date))
  a.sub = a[ a$Date >= rr[1] & a$Date <= rr[2],]
  b.sub = b[ b$Date >= rr[1] & b$Date <= rr[2],]
  structure(data.frame(a.sub$Date, 
                       a.sub$Adj.Close, 
                       b.sub$Adj.Close),
             names = c("Date", stockNames))
}

overlap = combine2Stocks(ss600036, ss600185)
names(overlap)

range(overlap$Date)


range(ss600036$Date)

range(ss600185$Date)


r = overlap$ss600036/overlap$ss600185

plotRatio =
function(r, k = 1, date = seq(along = r), ...)
{
  plot(date, r, type = "l", ...)
  abline(h = c(mean(r), 
               mean(r) + k * sd(r), 
               mean(r) - k * sd(r)), 
         col = c("darkgreen", rep("red", 2*length(k))), 
         lty = "dashed")
}
##################################
plot(density(r))
boxplot(r,main="格力")
plotRatio(r, k = 0.85, overlap$Date, col = "lightgray",
xlab = "Date", ylab = "Ratio")
#######################################
findNextPosition =
  # e.g.,  findNextPosition(r)
  #        findNextPosition(r, 1174)
  # Check they are increasing and correctly offset
function(ratio, startDay = 1, k = 1, 
          m = mean(ratio), s = sd(ratio))
{
  up = m + k *s
  down = m - k *s

  if(startDay > 1)
     ratio = ratio[ - (1:(startDay-1)) ]
    
  isExtreme = ratio >= up | ratio <= down
  
  if(!any(isExtreme))
      return(integer())

  start = which(isExtreme)[1]
  backToNormal = if(ratio[start] > up)
                     ratio[ - (1:start) ] <= m
                 else
                     ratio[ - (1:start) ] >= m

   # return either the end of the position or the index 
   # of the end of the vector.
   # Could return NA for not ended, i.e. which(backToNormal)[1]
   # for both cases. But then the caller has to interpret that.
   
  end = if(any(backToNormal))
           which(backToNormal)[1] + start
        else
           length(ratio)
  
  c(start, end) + startDay - 1 
}

k = .85
a = findNextPosition(r, k = k)

b = findNextPosition(r, a[2], k = k)

c = findNextPosition(r, b[2], k = k)

showPosition = 
function(days, ratios, radius = 20)
{
  symbols(days, ratios, circles = rep(radius, 2), 
           fg = c("darkgreen", "red"), add = TRUE, inches = FALSE)
}
#####################################################
#########################################################
showPosition =
function(pos, col = c("darkgreen", "red"), ...)
{
if(is.list(pos))
return(invisible(lapply(pos, showPosition, col = col, ...)))
abline(v = pos, col = col, ...)
}

abline(unlist(pos), col = col, ...)

showPosition =
function(pos, col = c("darkgreen", "red"), ...)
abline(v = unlist(pos), col = col, ...)

plotRatio(r, k, overlap$Date, xlab = "Date", ylab = "Ratio")
showPosition(overlap$Date[a], r[a])
showPosition(overlap$Date[b], r[b])
showPosition(overlap$Date[c], r[c])

####################################################
######################################################



getPositions =
function(ratio, k = 1, m = mean(ratio), s = sd(ratio))
{
   when = list()
   cur = 1

   while(cur < length(ratio)) {
      tmp = findNextPosition(ratio, cur, k, m, s)
      if(length(tmp) == 0)  # done
         break
      when[[length(when) + 1]] = tmp
      if(is.na(tmp[2]) || tmp[2] == length(ratio))
         break
      cur = tmp[2]
    }

   when
}

showPosition = 
function(days, ratio, radius = 5)
{
  if(is.list(days))
     days = unlist(days)

  symbols(days, ratio[days], 
          circles = rep(radius, length(days)), 
          fg = c("darkgreen", "red"),
          add = TRUE, inches = FALSE)
}
##############################################################
###############################################################

pos = getPositions(r, k)
plotRatio(r, k, overlap$Date, xlab = "Date", ylab = "Ratio")
invisible(lapply(pos, function(p)
showPosition(overlap$Date[p], r[p])))
k = .5
pos = getPositions(r, k)
plotRatio(r, k, col = "lightgray", ylab = "ratio")
showPosition(pos, r)
#################################################################
################################################################


positionProfit =
  #  r = overlap$att/overlap$verizon
  #  k = 1.7
  #  pos = getPositions(r, k)
  #  positionProfit(pos[[1]], overlap$att, overlap$verizon)
function(pos, stockPriceA, stockPriceB, 
         ratioMean = mean(stockPriceA/stockPriceB), 
         p = .001, byStock = FALSE)
{
  if(is.list(pos)) {
    ans = sapply(pos, positionProfit, 
                  stockPriceA, stockPriceB, ratioMean, p, byStock)
    if(byStock)
       rownames(ans) = c("A", "B", "commission")
    return(ans)
  }
    # prices at the start and end of the positions
  priceA = stockPriceA[pos]
  priceB = stockPriceB[pos]

    # how many units can we by of A and B with $1
  unitsOfA = 1/priceA[1]
  unitsOfB = 1/priceB[1]

    # The dollar amount of how many units we would buy of A and B
    # at the cost at the end of the position of each.
  amt = c(unitsOfA * priceA[2], unitsOfB * priceB[2])

    # Which stock are we selling
  sellWhat = if(priceA[1]/priceB[1] > ratioMean) "A" else "B"

  profit = if(sellWhat == "A") 
              c((1 - amt[1]),  (amt[2] - 1), - p * sum(amt))
           else 
              c( (1 - amt[2]),  (amt[1] - 1),  - p * sum(amt))

  if(byStock)
     profit
  else
     sum(profit)
}

pf = positionProfit(c(1, 2), c(3838.48, 8712.87), 
                             c(459.11, 1100.65), p = 0)
pos = getPositions(r, k)
prof = positionProfit(pos, overlap$ss110030, overlap$ss600185, mean(r))

summary(prof)


i = 1:floor(nrow(overlap)/2)
train = overlap[i, ]
test = overlap[ - i, ]

r.train = train$ss110030/train$ss600185
r.test = test$ss110030/test$ss600185

k.max = max((r.train - mean(r.train))/sd(r.train))

k.min = min((abs(r.train - mean(r.train))/sd(r.train)))

ks = seq(k.min, k.max, length = 1000)
m  = mean(r.train)


profits =
 sapply(ks,
        function(k) {
           pos = getPositions(r.train, k)
           sum(positionProfit(pos, train$ss110030, train$ss600185, 
                               mean(r.train)))
        })

 

plot(ks,profits,type="l",xlab="k",ylab="Profit")

ks[  profits == max(profits) ] 

tmp.k = ks[  profits == max(profits) ]  
pos = getPositions(r.train, tmp.k[1])
all(sapply(tmp.k[-1],
            function(k) 
               identical(pos, getPositions(r.train, k))))

k.star = mean(ks[  profits == max(profits) ]  )
#k.star =0.85
pos = getPositions(r.test, k.star, mean(r.train), sd(r.train))
testProfit = sum(positionProfit(pos, test$ss110030, test$ss600185))   

#################################################################
##################################################################

max_local=pos[[length(pos)]]
overlap[max_local,]
is.na(overlap[max_local,]$Date[1])==FALSE&&is.na(overlap[max_local,]$Date[2])==TRUE
if(is.na(overlap[max_local,]$Date[1])==FALSE
   &&is.na(overlap[max_local,]$Date[2])==TRUE)#??һ?????ֻؿ??ֵ??жϴ???????
{
write.csv(file = "stocksToEnter.csv",
          data.frame(list(sym = colnames(overlap)[2], type = "buy",
                          OrderPrice="?м?",OrderVolume=1000)))
}
if(is.na(overlap[max_local,]$Date[1])==FALSE
   &&is.na(overlap[max_local,]$Date[2])==FALSE)#??һ?????ֻ?ƽ?ֵ??жϴ???????
{
  write.csv(file = "stocksToExit.csv",
            data.frame(list(sym = colnames(overlap)[2], type = "sale",
                            OrderPrice="?м?",OrderVolume=1000)))
}
##############################################################
###############################################################

plotRatio(r.test, k.star, test$Date, xlab = "Date", ylab = "Ratio")
invisible(lapply(pos, function(p)
  showPosition(test$Date[p], r[p])))

plotRatio(r.test, k.star, col = "lightgray", ylab = "ratio")
showPosition(pos, r.test)
#################################################################
################################################################




stockSim = 
function(n = 4000, rho = 0.99, psi = 0, sigma = rep(1, 2),
         beta0 = rep(100, 2), beta1 = rep(0, 2),
         epsilon = matrix(rnorm(2*n, sd = sigma),
                           nrow = n, byrow = TRUE))
{
  X = matrix(0, nrow = n, ncol = 2)
  X[1,] = epsilon[1,]

  A = matrix(c(rho, psi*(1-rho), psi*(1-rho), rho), nrow = 2)
  for(i in 2:n)
      X[i,] = A %*% X[i-1,] + epsilon[i,]
  
       # Add in the trends, in place
  X[,1] = beta0[1] + beta1[1] * (1:n) + X[,1]
  X[,2] = beta0[2] + beta1[2] * (1:n) + X[,2]

  X
}

set.seed(12312)

a = stockSim(rho = .99, psi = 0)



a = stockSim(beta1 = c(.05, .1))

a =stockSim(rho = .99, psi = 0,beta1 = c(.05, .1))

matplot(1:nrow(a), a, type = "l", xlab = "Day", ylab = "Y",
        col = c("black", "grey"), lty = "solid")

runSim = 
function(rho, psi, beta0 = c(100, 100), beta1 = c(0, 0),
         sigma = c(1, 1), n = 4000)
{
    X = stockSim(n, rho, psi, sigma, beta = beta0, beta1 = beta1)
    train = X[ 1:floor(n/2), ]
    test = X[ (floor(n/2)+1):n, ]
    m = mean(train[, 1]/train[, 2])
    s = sd(train[, 1]/train[, 2])
    k.star = getBestK(train[, 1], train[, 2], m = m, s = s)
    getProfit.K(k.star, test[, 1], test[, 2], m, s)
}

getProfit.K =
function(k, x, y, m = mean(x/y), s = sd(x/y))
{
    pos = getPositions(x/y, k, m = m, s = s)
    if(length(pos) == 0)  
       0
    else
       sum(positionProfit(pos, x, y, m))
}

getBestK = 
function(x, y, ks = seq(0.1, max.k, length = N), N = 100, 
         max.k = NA, m = mean(x/y), s = sd(x/y))
{
    if(is.na(max.k)) {
       r = x/y
       max.k = max(r/sd(r))
    }

    pr.k = sapply(ks, getProfit.K, x, y, m = m, s = s)
    median(ks[ pr.k == max(pr.k) ])
}

simProfitDist = 
function(..., B = 999) 
      sapply(1:B,  function(i, ...) runSim(...), ...)

#set.seed(1223)
set.seed(c(403L, 480L, 1186868935L, -564132622L, 102911975L, 237616004L, 
1116051066L, -1043128591L, -1460313398L, -1395138954L, -365778259L, 
-1225485068L, 1718071315L, -649192418L, -921342130L, 346075150L, 
1722654649L, -327604396L, 244121873L, -700634160L, 613508071L, 
804493508L, -507633940L, 1488427131L, 213676733L, 1687978272L, 
-1127251234L, 1382197899L, -1022807050L, -1864259465L, 885445730L, 
1290386470L, -1331261490L, -999287442L, -277092293L, 2101408359L, 
-1456661848L, -785692150L, 72441051L, 1365767073L, -1294219473L, 
1017186604L, -1116706053L, -2068003772L, -165380810L, -875700181L, 
1224089016L, 480405670L, -252200051L, -623491476L, -1242047912L, 
806518670L, -1947064534L, -372082416L, -579138906L, 108336168L, 
1139002183L, -143363498L, 593294083L, -864469916L, -1869706882L, 
903880706L, 386302804L, 809820110L, 133985801L, 79699708L, 688647927L, 
-943750890L, 1163299363L, -807932427L, -944789667L, 1920872146L, 
-921910240L, -131302831L, 530329613L, 1827118648L, 92807878L, 
2065900764L, -1218936651L, 1181349252L, 67299171L, 2074836131L, 
-597280012L, -1263713799L, 1548031089L, 1715306840L, -852221040L, 
-1547852301L, -527879295L, -436935652L, 1924022897L, -842862842L, 
-2077935048L, 1356350596L, 1870611013L, -2103474623L, -1613679300L, 
166792814L, -1102812317L, 2077897590L, 480518401L, 379090865L, 
-26610149L, 656649370L, -1761591953L, -1254708659L, 2121846031L, 
1438051445L, -1304860000L, -188112113L, -1824487043L, 1581331000L, 
1029113458L, 1902310176L, -1936760301L, 396851249L, -1207829603L, 
-1913978508L, -20213853L, -209477658L, 499625901L, 2031567202L, 
-1543728792L, -1706768382L, 136724520L, -347740650L, 483157824L, 
-236106890L, -1500586317L, 453024740L, 1703504442L, -263880589L, 
-125112602L, -1001997840L, -1592424975L, -773553734L, 803886903L, 
-115132810L, 1610947798L, 1824663828L, 838102973L, -1058649224L, 
-1051856131L, 961990765L, 573384990L, 1228840248L, -1996598967L, 
1182020075L, -434153999L, 168816796L, 1420025005L, 1200209823L, 
1264985465L, -626121664L, 620352090L, -1315932826L, -843331419L, 
-957337700L, 914648676L, -829005324L, -1319589280L, 122059527L, 
-299647738L, -943841569L, 866528602L, -1027466074L, -126553854L, 
1547026185L, -1681425701L, -1933246053L, -417507122L, 2068710278L, 
1989903281L, -1347367358L, 1902101510L, 1731911669L, 1289996204L, 
-1250271848L, -1879133788L, 722008977L, -427662154L, 1413873183L, 
-1169622555L, 762213446L, 1667599507L, -1666566676L, 1652684703L, 
350435036L, -2033705484L, 2097849370L, -1630232896L, 2025340231L, 
-1063116874L, -1314061427L, -1144581189L, -64190072L, 1663791892L, 
1883026955L, 153863078L, -402877850L, -875252159L, -1930523042L, 
18406843L, 329918134L, 1290215988L, -979623579L, 344430536L, 
-1143865619L, -340016464L, -458102918L, 517646039L, 1231142666L, 
166528805L, -342193758L, 904615585L, -1205515778L, 724715941L, 
968127782L, 243538646L, 1979375126L, 1325562739L, 1340888866L, 
355336170L, -17023070L, 344367027L, -67118898L, 1326652915L, 
1625731497L, 147433265L, -830093249L, 1993908517L, 821064902L, 
-762989438L, -1440996861L, -196858430L, -1238089535L, 1673543940L, 
1892872693L, -1051775329L, -282328944L, 1336333588L, -212690986L, 
1386349395L, -729248688L, 1642308976L, 1646463749L, -1419274305L, 
-1826645083L, -1507200178L, -366202243L, -1044756705L, -1159227266L, 
1113476239L, -3424355L, -185755738L, -916681244L, 1324957768L, 
73351117L, -720017563L, 1591002622L, -1852591269L, 1652154449L, 
-1665742768L, -68675807L, -1160700221L, 768415276L, 325177830L, 
67627886L, -1454697091L, 882610425L, -1814789550L, 1555321478L, 
753275225L, 1464696247L, 627076189L, 1339905647L, 624743940L, 
-1185620908L, -2146266305L, -938656926L, 581335881L, 189184346L, 
218385640L, -989832929L, 500674659L, 393584110L, -794092778L, 
-1747312242L, 341177893L, -772854010L, 1475517398L, -1590682904L, 
-951069287L, -582008776L, -393901990L, 1168586772L, -1963543282L, 
-974514217L, 1020576517L, -764213959L, -1834023646L, -1747661644L, 
-1641003649L, 795957625L, 1184412202L, 1812449494L, -1126769082L, 
2043773316L, 1802040463L, 1715846278L, 1842272519L, 1450224806L, 
-1511654170L, -1197989434L, 1978601803L, 610799714L, 1088668469L, 
-1902095144L, 1915111009L, 1478008839L, -1581614541L, -1339626468L, 
441923512L, -31782675L, -89725413L, 1411278985L, -1755216132L, 
-795622722L, -370367419L, -1385048526L, -2091182238L, 109838800L, 
-1070895461L, -1498858252L, -1889097556L, 1898445905L, -1621811940L, 
-954231543L, -137556740L, -989248721L, 105296762L, -1996839662L, 
785916371L, -616080183L, 1525155691L, -833424094L, 2100843589L, 
-1555984385L, -80030735L, -1250883040L, 1831433773L, 547578515L, 
-810629343L, -1142726467L, 2117321962L, -2140596793L, 1313632703L, 
1267190798L, -68661364L, -6903384L, 1151045236L, -1729352474L, 
1000291571L, 1268115103L, -1444396042L, -2138118203L, 1617430145L, 
-1682932503L, 619890163L, 180728650L, -644618767L, -797263264L, 
-2047172487L, 754698436L, 222298246L, 1517236576L, 5693007L, 
-595480998L, 1632494212L, -1424083183L, 1412050369L, 2142250409L, 
-602833022L, -403699501L, -490827995L, 1557574953L, -2132444272L, 
-55639357L, 1019965591L, 139069060L, -1657892388L, -767965493L, 
576040048L, -1548340030L, 1131284195L, 1336435339L, -1905034547L, 
-730149305L, 1618894598L, 71584500L, 1423629480L, 1669188782L, 
-685187439L, 1272690885L, -553379344L, -545264842L, 2095555942L, 
-83035847L, 96305956L, 639612902L, 577702268L, 1677338672L, -942419268L, 
420759515L, -1325562341L, 985821259L, -525187086L, 520539173L, 
433751150L, -2118298443L, 719833059L, -174473153L, -738019559L, 
1703250361L, 524044374L, -1045820881L, -1804959311L, -832929714L, 
2047917224L, -1247781262L, 818725770L, -1714624847L, -2016647132L, 
-189168645L, 286479090L, 1530347634L, -329440666L, 739073031L, 
-1897583738L, -1662281093L, -900174567L, -45647943L, 414561875L, 
1041731495L, -376974013L, 51580041L, 384733270L, 2054144600L, 
-1312109074L, 548330855L, -797034814L, -1370281463L, -541739557L, 
286777741L, 1494225816L, 1305011062L, 163439479L, -1316886918L, 
-308302580L, 49829526L, 910209580L, -841223820L, -858224003L, 
143888201L, -810350237L, 1507716383L, -2034818598L, 660495953L, 
-1655211060L, 925267207L, -2052138769L, 229359900L, 1888017947L, 
1576043327L, -33350256L, -1274782820L, 1005309164L, 1614843993L, 
1264273356L, -1528924689L, -304582542L, -121638437L, 1171406801L, 
736971169L, -2146631505L, -1743824538L, -614198349L, 943701939L, 
-2030892395L, 1516639311L, -572299007L, 239562249L, -1889976915L, 
929753257L, 1805105579L, 148221925L, 897909266L, -1349508457L, 
-643312147L, 468967691L, -1469484917L, 1768806359L, 1082713852L, 
15256142L, -1086558832L, -432362074L, -1647681512L, 1514855277L, 
417942991L, -2133276866L, 1408915913L, 296386833L, -1315661524L, 
-297528064L, -1786227800L, -2091447162L, -768659080L, -1177928134L, 
-1864369632L, 407421047L, -1859244770L, -1076724520L, -376740377L, 
878753555L, -664340467L, 832172489L, -836963091L, 848332253L, 
-1807334490L, 2098036300L, -1109042254L, -1174741419L, 1184700053L, 
968620968L, -525614987L, -114793041L, -1321919532L, -1595484247L, 
1600333783L, 29908102L, -715949684L, 583357370L, 120037556L, 
-335509067L, 1737920869L, -1241624930L, 1606786862L, -737548562L, 
-1611944436L, 1656419288L, -718120537L, -451010933L, -110285493L, 
1292547266L, -1089551195L, -1637594082L, -1441928276L, -1705968635L, 
-2029704021L, -162727542L, -1656395814L, -1469050535L, 612066363L, 
-572366400L, -1553370828L, 702526565L, -639237497L, 187979532L, 
155187202L, 1641064685L, -122338331L, 1378118844L, 878877783L, 
1205191245L, -1750912513L, -924627532L, -309600607L, -681044640L, 
-1963054540L, 1320942754L, 925034724L, 818207978L, 361092738L, 
1421454926L, -566990230L, -1701907548L, 201363241L, -1809384850L, 
-2114571448L, -529351149L, 1843691077L, 2008461407L, -1308347844L, 
155503773L, 301465246L, 1858949722L, -205612146L, -790706025L, 
1036695590L, 510649891L, -201008936L, 2015640535L, -716758454L, 
358462269L, -1009276902L, -1126776832L, 507114889L, -1711422687L, 
-510277455L, -110784271L, 1798988236L, -427992591L, -539979934L, 
-719340882L, -1620994430L, -1482620310L, 685700851L, -1546350852L, 
-2044066697L, -485689634L, -32397641L, 822280247L, -1768970340L, 
1370989923L, 1664628646L, -634783558L))

system.time({ x = simProfitDist( .99, .9, c(0, 0)) })

summary(x)

plot(density(x))

sum(x < 0)/length(x)

plog = expand.grid(psi = seq(.8, .99, length = 20),  
                beta1 = seq(-.01, .01, length = 20),  
                beta2 = seq(-.01, .01, length = 20))

Rprof("sim.prof")
system.time({x = simProfitDist( .99, .9, c(0, 0))})
Rprof(NULL)
head(summaryRprof("sim.prof")$by.self)


counter = 0L
trace(findNextPosition, quote( counter <<- counter + 1L), 
         print = FALSE)

system.time({x = simProfitDist( .99, .9, c(0, 0))})

untrace(findNextPosition)

library(compiler)
stockSim.cmp = cmpfun(stockSim)

tm.orig = system.time({replicate(80, stockSim())})
tm.compiled = system.time({replicate(80, stockSim.cmp())})
tm.orig/tm.compiled

dyn.load("stockSim.dll")

stockSim.c = 
function(n = 4000, rho = 0.99, psi = 0, sigma = rep(1, 2),
         beta0 = rep(100, 2), beta1 = rep(0, 2),
         epsilon = matrix(rnorm(2*n, sd = sigma), nrow = n))
{
  X = matrix(0, nrow = n, ncol = 2)
  X[1,] = epsilon[1,]
  X = .C("stockSim", X, as.integer(n), rho, psi, epsilon)[[1]]
  
       # Add in the trends
  X[,1] = beta0[1] + beta1[1] * (1:n) + X[,1]
  X[,2] = beta0[2] + beta1[2] * (1:n) + X[,2]

  X
}

e = matrix(rnorm(2*4000, sd = c(1, 1)), , 2)
tmp1 = stockSim.c(epsilon = e)
tmp2 = stockSim(epsilon = e)
identical(tmp1, tmp2)

stockSim = stockSim.c

Rprof("sim.prof")
system.time({x = simProfitDist( .99, .9, c(0, 0))})
Rprof(NULL)
head(summaryRprof("sim.prof")$by.self)

p = seq(.5, 1, length = 20)
params = as.matrix(expand.grid(p, p))

profits = apply(params, 1, 
                function(p) 
                   median(simProfitDist(p[1], p[2], B = 100)))
