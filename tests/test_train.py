from smp.plot_history import plot_logs
from main import main


def test_train():
    main()


def test_plot():
    plot_logs(show=False)
