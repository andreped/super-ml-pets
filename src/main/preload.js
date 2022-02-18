window.addEventListener("DOMContentLoaded", () => {
    const replaceText = (selector, text) => {
        const element = document.getElementById(selector);
        if (element) element.innerText = text;
    };

    replaceText("status", "success");
});

$.ajax({
    type: "POST",
    url: "~/pythoncode.py",
    data: { param: text },
}).done(function (o) {
    // do something
});
