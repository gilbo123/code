import Vue from 'vue'
import VueRouter from 'vue-router'
// import VueFrappe from 'vue2-frappe';
import Chart from 'vue2-frappe'
import App from './App.vue'
import Pie from './components/Pie.vue'
import Bar from './components/Bar.vue'
import Weights from './components/Weights.vue'
import Home from './components/Home.vue'


//Charts
Vue.use(Chart)
Vue.use(VueRouter)

Vue.config.productionTip = false

const routes = [
  { path: '/', component: Home },
  { path: '/bar', component: Bar },
  { path: '/weights', component: Weights },
  { path: '/pie', component: Pie }
]
const router = new VueRouter({
  routes
})


new Vue({
  router,
  render: h => h(App),
}).$mount('#app')
