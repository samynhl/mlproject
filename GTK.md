### To initiazlize an empty github repository
```
git init
```
---
### Link your github repository to your remote repository
```
git remote add origin https://github.com/samynhl/mlproject.git
```
---
### Check your remote coonnection links (fetch and push)
```
git remote -v
```
---
### Check username and email
```
git config --global user.email
```
---
### You better use the Python .gitignore file, it contains nearly everything
---
## setup.py
- responsible for creating the ml project as a package and even deploy in Pypi
- The "-e ." in requirements file allows you to automatically trigger setup.py