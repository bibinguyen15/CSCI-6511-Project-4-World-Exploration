import requests
import json
from constants import *


headers = {
    'User-Agent': 'PostmanRuntime/7.32.2',
    'Accept': '*/*',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'close',
    'Content-Type': 'application/x-www-form-urlencoded',
    'x-api-key': xapikey,
    'userId': userId,
}


# Operations for getting world information
def getLocation():
    params = {
        'type': 'location',
        'teamId': teamId
    }

    response = requests.request(
        "GET", WORLD_URL, headers=headers, params=params)

    jsonData = json.loads(response.text)

    print("getLocation():", jsonData)
    return jsonData


def enterWorld(world=0):
    payload = {'type': 'enter',
               'worldId': world,
               'teamId': teamId}
    response = requests.request(
        "POST", WORLD_URL, headers=headers, data=payload)

    jsonData = json.loads(response.text)

    print("enterWorld():", jsonData)

    if jsonData['code'] == 'FAIL':
        return jsonData['message']

    else:
        return jsonData


def makeMove(world, move):
    payload = {'type': 'move',
               'teamId': teamId,
               'move': move,
               'worldId': world}

    response = requests.request(
        "POST", WORLD_URL, headers=headers, data=payload)

    jsonData = json.loads(response.text)

    print(jsonData)

    # if jsonData['code'] == 'FAIL':
    # return jsonData['message']

    # else:
    # return jsonData
    return jsonData


# Operations regarding team
def getRuns(count=1):
    params = {
        'type': 'runs',
        'teamId': teamId,
        'count': count
    }

    response = requests.request(
        "GET", TEAM_URL, headers=headers, params=params)

    jsonData = json.loads(response.text)
    print(jsonData)


def getScore():
    params = {
        'type': 'score',
        'teamId': teamId
    }

    response = requests.request(
        "GET", TEAM_URL, headers=headers, params=params)

    jsonData = json.loads(response.text)
    print(jsonData)


# Reseting a team
def resetTeam():
    params = {
        'teamId': teamId,
        'otp': OTP
    }

    response = requests.request(
        "GET", RESET_URL, headers=headers, params=params)

    jsonData = json.loads(response.text)

    print(jsonData)
    return jsonData['code']


# getScore()
# getRuns()
# getLocation()
resetTeam()

# enterWorld(0)
# Response: {'code': 'OK', 'worldId': 0, 'runId': 44176, 'state': '0:0'}

# makeMove(0, 'N')
# Response: {'code': 'OK', 'worldId': 0, 'runId': '44176', 'reward': -0.1, 'scoreIncrement': -0.1, 'newState': {'x': '0', 'y': 1}}
