
```shell
nohup python3 manage.py runserver 5556 > /dev/null 2>&1 &
gunicorn -c gunicorn.conf.py CustomerService.wsgi:application
```