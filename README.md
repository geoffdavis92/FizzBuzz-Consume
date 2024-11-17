# FizzBuzz-Consume
A project that consumes a FizzBuzz model and predicts possible FizzBuzz result
![image](https://github.com/user-attachments/assets/e4911fd8-4a7b-4776-97cf-97f4cd268ca8)


## How to run
```
npm install
npm start
```

## Note on Node Version
If you encounter an error running on macOS, you can either use a version manager like `nvm` or you can allow legacy OpenSSL by updating the package.json script to allow legacy OpenSSL.

```
  "scripts": {
    "start": "NODE_OPTIONS='--openssl-legacy-provider' react-scripts start",
    "build": "react-scripts build",
    "test": "react-scripts test --env=jsdom",
    "eject": "react-scripts eject"
  },
```