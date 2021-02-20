import configparser


def main():
    config = configparser.ConfigParser()
    config.read("config.yml")
    pass

if __name__ == "__main__":
    main()
