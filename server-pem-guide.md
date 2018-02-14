
### 01 Email

---

안녕하세요,

아래와 같이 p2.xlarge instance 1개를 생성하였습니다. 접속에 필요한 정보는 아래와 같습니다.
IP: 13.125.75.97
login user: ubuntu
password: 첨부한 .pem 파일 참고

AWS는 ID/PW 방식이 아닌 private key 방식으로 접속할 수 있습니다. 첨부한 파일 및 아래 AWS 접속 방법을 통해 접속할 수 있습니다.
AWS 접속 방법 링크: http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AccessingInstances.html
ubuntu 사용자는 sudoer이므로 'sudo' 명령어를 통해 root 권한을 획득할 수 있습니다.
보안 상의 문제로 22번, 8000~8100번 포트만 외부에서 접근할 수 있습니다. 혹시 다른 포트 개방도 필요하면 메일로 연락주세요.
사용 중 다른 문제 및 궁금한 사항이 있으면 메일로 연락주세요.

감사합니다.
이주현 드림.


---

### 02 처음 환경설정

+ pem 다운로드
+ pem 권한설정

`chmod 400 pem-path`

+ 서버 접속

`ssh ubuntu@13.125.75.97 -i <path:/home/snu/jmc/AI-04.pem>`

`ssh ubuntu@13.125.75.97 -i /home/snu/jmc/AI-04.pem`

+ **현재 프로세스 상태 확인**

`sudo nvidia-docker ps -a`


+ 처음 process 실행

`sudo nvidia-docker start <name:cnn_nlp>`

---

### 03 작업

+ docker process attach (enter the command below and press enter)

`sudo nvidia-docker attach <prcoess_name:cnn_nlp>`

+ docker process detach (ctrl+p ctrl+q)

> **Note**: docker: 컴퓨터, tmux: 프로세스 (ctrl+b ctrl+d)

+ tmux

`@@@resume`

+ 파일 전송

`scp /home/snu/jmc/snufira-aws/data ubunutu@@13.125.75.97:/home`

---
