# hdiyf-backend

## Features

- get access to a random news
- (wip) : predict if a news is real or fake

## Installation

Create a new virtual environment with python 3.8x, then install the dependencies :
```
pip install -r requirements.txt
```

## How to use

### Locally

```
# Run the app
$ uvicorn app.main:app

# Go to localhost:8000/api/v1/docs
```

### Cloud

### Setup

```
# Upload new database
$ heroku pg:psql DATABASE_URL --app hdiyf-backend < hdiyf_database.dump
```

#### Access to doc online

https://hdiyf-backend.herokuapp.com/api/docs#/