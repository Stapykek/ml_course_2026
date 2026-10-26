import util
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def main():
    do_linear_regression("lab1_DS.csv", x_col='YearsExperience', y_col='Salary')
    return


def do_linear_regression(filepath: str, x_col: str, y_col: str, describe_only_xy: bool = True) -> None:
    # загрузка датасета
    data = util.read_file(filepath)

    if x_col not in data.columns:
        raise ValueError('В датасете нет столбца x:', x_col)

    if y_col not in data.columns:
        raise ValueError('В датасете нет столбца y:', y_col)

    # описание датасета
    print("Статистическое описание датасета:")
    print(util.get_statistic_description(data[[x_col, y_col]] if describe_only_xy else data))

    # создание изображения из трёх графиков
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    x = data[x_col].to_numpy()
    y = data[y_col].to_numpy()

    # решение задачи линейной регрессии методом наименьших квадратов.
    model = util.LinearRegressionModel()
    model.fit(x, y)

    print("Параметры модели y = wx + b:")
    print("w =", model.w)
    print("b =", model.b)
    y_pred = model.predict(x)

    max_x = max(x)
    max_y = max(max(y), max(y_pred))

    # построение графиков
    x_lower_lim = 0
    x_upper_lim = max_x * 1.1
    y_lower_lim = 0
    y_upper_lim = max_y * 1.1
    x_ticksize = max_x/11
    y_ticksize = max_y/11

    # # график 1 только с точками
    ax1.scatter(x, y, label='истинные точки')

    ax1.legend(loc='upper left')

    ax1.set_title('Исходные точки')
    ax1.set_xlabel('Опыт, лет')
    ax1.set_ylabel('Зарплата')
    ax1.set_xlim(x_lower_lim, x_upper_lim)
    ax1.set_ylim(y_lower_lim, y_upper_lim)

    # # график 2 с точками датасета и регрессионной прямой
    ax2.scatter(x, y, label="истинные точки")

    ax2.plot(x, y_pred, color='red', label="предсказания линейной регрессии")

    ax2.legend(loc='upper left')

    ax2.set_title('Предсказание')
    ax2.set_xlabel('Опыт, лет')
    ax2.set_ylabel('Зарплата')
    ax2.set_xlim(x_lower_lim, x_upper_lim)
    ax2.set_ylim(y_lower_lim, y_upper_lim)

    # # график 3 с точками датасета, регрессионой прямой и визуализацией площадей квадратов.
    for i in range(len(x)):
        side_len = abs(y[i] - y_pred[i])/y_ticksize
        rect = patches.Rectangle(
            (x[i], min(y[i], y_pred[i])), side_len*x_ticksize, side_len*y_ticksize,
            linewidth=1,
            edgecolor='black',
            facecolor='white',
            hatch='////'
        )
        ax3.add_patch(rect)

    ax3.scatter(x, y, label="истинные точки")

    ax3.plot(x, y_pred, color='red', label="предсказания линейной регрессии")

    ax3.legend(loc='upper left')

    ax3.set_title('Визуализация площадей квадратов')
    ax3.set_xlabel('Опыт, лет')
    ax3.set_ylabel('Зарплата')
    ax3.set_xlim(x_lower_lim, x_upper_lim)
    ax3.set_ylim(y_lower_lim, y_upper_lim)

    fig.suptitle("Метод наименьших квадратов", fontsize=20)
    plt.show()


if __name__ == "__main__":
    main()
