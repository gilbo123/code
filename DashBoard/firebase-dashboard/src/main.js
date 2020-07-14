import Vue from 'vue'
import VueRouter from 'vue-router'
// import VueFrappe from 'vue2-frappe';
import Chart from 'vue2-frappe'
import App from './App.vue'
import Pie from './components/Pie.vue'
import Bar from './components/Bar.vue'
import Home from './components/Home.vue'

Vue.use(Chart)
Vue.use(VueRouter)

Vue.config.productionTip = false

//Routes
// const Home  = {
//   template: '<div>Home</div>'
// }
// const Bar  = {
//   template: '<div>Bar</div>'
// }
// const Pie  = {
//   template: '<div>Pie</div>'
// }
const HomePage = Home
const BarChart = Bar
const PieChart = Pie

const routes = [
  { path: '/', component: HomePage },
  { path: '/bar', component: BarChart },
  { path: '/pie', component: PieChart }
]
const router = new VueRouter({
  routes
})


new Vue({
  router,
  render: h => h(App),
}).$mount('#app')
