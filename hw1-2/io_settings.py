SETTINGS_PATH = 'settings.txt'


def read_settings(path):
    settings = {}
    file_settings = open(path, 'r')
    for line in file_settings:
        key, value = (x.strip() for x in line.strip().split(' '))
        settings[key] = int(value)
    return settings


def write_settings(path, settings):
    file_settings = open(path, 'w')
    for key, value in settings.items():
        file_settings.write(str(key) + ' ' + str(value) + '\n')


if __name__ == '__main__':
    settings = read_settings(SETTINGS_PATH)
    print(settings)
    write_settings(SETTINGS_PATH, settings)