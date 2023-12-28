const $container = document.querySelector(".container");
const $main = $container.querySelector(".contents");
const $btnFold = document.querySelector(".btn-fold");
const $nav = $main.querySelector(".question-nav");
const $codeMirror = document.querySelector(".CodeMirror");
const $btnMenu = $container.querySelector(".hamburger-btn");
const $icon = $btnMenu.querySelector("img");
const $menuContainer = $container.querySelector(".menu-list");
let isMobile = null;

const handleCloseQuestions = () => {
    $container.classList.remove("menu-on");
    $btnFold.innerText = "메뉴 펼치기";
    $codeMirror.classList.remove("menu-on-CodeMirror");
    $codeMirror.classList.add("menu-off-CodeMirror");
};

const handleOpenQuestions = () => {
    if ($menuContainer.classList.contains("is-active")) {
        handleCloseMenu();
    }

    $container.classList.add("menu-on");
    $btnFold.innerText = "메뉴 접기";
    $codeMirror.classList.add("menu-on-CodeMirror");
    $codeMirror.classList.remove("menu-off-CodeMirror");
};

const handleToggleQuestions = () => {
    if ($container.classList.contains("menu-on")) {
        handleCloseQuestions();
    } else {
        handleOpenQuestions();
    }
};

const handleToggleMenu = () => {
    let isActive = $btnMenu.classList.contains("is-active");

    $btnMenu.classList.toggle("is-active");
    $icon.src = isActive ? "src/img/hamburger-btn.webp" : "src/img/close.webp";
    $icon.alt = isActive ? "메뉴 열기" : "메뉴 닫기";

    if (isActive) {
        $icon.classList.remove("close");
        $menuContainer.classList.remove("active");
    } else {
        $icon.classList.add("close");
        $menuContainer.classList.add("active");
    }

    // if ($menuContainer.classList.contains("is-active")) {
    //     handleCloseQuestions();
    // }
};

const handleCloseMenu = () => {
    $menuContainer.classList.remove("active");
    $icon.classList.remove("close");
    $icon.src = "src/img/hamburger-btn.webp";
};

const checkMobile = () => {
    const winWidth = window.innerWidth;

    if (winWidth > 1024) {
        isMobile = false;
        $menuContainer.classList.remove("is-active");
    } else if (winWidth >= 900) {
        handleCloseMenu();
        isMobile = true;
    } else {
        handleCloseQuestions();
        isMobile = true;
    }
};

const handleResizeWidth = () => {
    let timer = null;
    clearTimeout(timer);
    timer = setTimeout(function () {
        checkMobile();
    }, 300);
};

const handleCloseMobileQuestions = (e) => {
    const check =
        Boolean(e.target.closest("nav")) || Boolean(e.target.closest("header"));
    if (isMobile && $container.classList.contains("menu-on") && !check) {
        handleCloseQuestions();
    }
};

const handleClickOutside = (e) => {
    if ($main.contains(e.target)) {
        handleCloseMenu();
    }
};

checkMobile();
$btnFold.addEventListener("click", handleToggleQuestions);
$container.addEventListener("click", handleCloseMobileQuestions);
window.addEventListener("resize", handleResizeWidth);

$btnMenu.addEventListener("click", handleToggleMenu);
$main.addEventListener("click", handleClickOutside);
