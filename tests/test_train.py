from smp.plot_history import plot_logs
from main import main


@pytest.fixture
def test_plot():
    plot_logs()


@pytest.mark.usefixtures("test_plot")
def test_train():
    main()
