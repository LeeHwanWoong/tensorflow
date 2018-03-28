import requests
import openpyxl
from bs4 import BeautifulSoup


movie_num = 98438
row_index = 1
f = open("test2.txt",'w')
f.close()
for i in range(3):
	movie_num = movie_num+i

	req = requests.get('https://movie.naver.com/movie/bi/mi/pointWriteFormList.nhn?code='+str(movie_num)+'&type=after&isActualPointWriteExecute=false&isMileageSubscriptionAlready=false&isMileageSubscriptionReject=false&page=1')
	html = req.text
	soup = BeautifulSoup(html,'html.parser')
	try :
		num = soup.select(
			'div.score_total > strong > em'
			)
		num = str(num)
		num = num.split('>')[1]
		num = num.split('<')[0]
		num = num.replace(',','')
		num = int(num)
	except :
		continue

	for i in range(num/10):
		i+=1
		req = requests.get('https://movie.naver.com/movie/bi/mi/pointWriteFormList.nhn?code='+str(movie_num)+'&type=after&isActualPointWriteExecute=false&isMileageSubscriptionAlready=false&isMileageSubscriptionReject=false&page='+str(i))
		html = req.text
		soup = BeautifulSoup(html,'html.parser')


		title = soup.select(
			'div.score_reple > p'
			)
		score = soup.select(
			'div.star_score > em'
			)
		a = []
		b = []
		for i in score:
			a.append(str(i))
		for i in title:
			b.append(str(i))
		i = 0
		j = 0
		for k in range(10):
			f = open("test.txt",'a')

			a[i] = a[i][4:]
			a[i] = a[i].split('<')
			b[i] = b[i][3:]
			if '</span>' in b[i]:
				b[i] = b[i].split('</span>')
				b[i] = b[i][1]
			b[i] = b[i].split('<')
			f.write(str(a[i][0]) + "\t" + str(b[i][0])+'\n')
			i=i+1
			j=j+1
			f.close()