const path = location.pathname.split("/").pop() || "index.html"

document.querySelectorAll("[data-nav]").forEach((a) => {
  const href = a.getAttribute("href")
  if (href === path) a.dataset.active = "true"
})

